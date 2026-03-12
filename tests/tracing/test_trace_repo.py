from __future__ import annotations

from datetime import timedelta
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException, MlflowTracingException
from mlflow.tracing.trace_repo import (
    TraceArchiveData,
    _validate_archive_params,
    archive_traces,
    delete_traces,
    load_archived_spans,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_local_store(**overrides):
    store = mock.MagicMock()
    store.ManagedSessionMaker = mock.MagicMock()
    store.collect_archive_candidates = mock.MagicMock(return_value=[])
    store.read_trace_for_archive = mock.MagicMock(return_value=None)
    store.mark_trace_archived = mock.MagicMock()
    store.get_trace_repository_artifact_uri = mock.MagicMock(return_value=None)
    store.find_archived_trace_uris = mock.MagicMock(return_value={})
    store.delete_trace_rows = mock.MagicMock(return_value=0)
    for k, v in overrides.items():
        setattr(store, k, v)
    return store


def _make_rest_store(**overrides):
    store = mock.MagicMock(spec=["archive_traces", "delete_traces"])
    store.archive_traces = mock.MagicMock(return_value=0)
    store.delete_traces = mock.MagicMock(return_value=0)
    for k, v in overrides.items():
        setattr(store, k, v)
    return store


def _make_archive_data(trace_id="t1", experiment_id=1):
    return TraceArchiveData(
        trace_id=trace_id,
        experiment_id=experiment_id,
        spans=[mock.MagicMock()],
        artifact_uri=f"file:///repo/{experiment_id}/traces/{trace_id}",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_requires_at_least_one_param():
    with pytest.raises(MlflowException, match="older_than or trace_ids"):
        _validate_archive_params(None, None)


def test_validate_older_than_must_be_positive():
    with pytest.raises(MlflowException, match="older_than must be positive"):
        _validate_archive_params(timedelta(0), None)


def test_validate_valid_params_pass():
    _validate_archive_params(timedelta(days=30), None)
    _validate_archive_params(None, ["tid-1"])


# ---------------------------------------------------------------------------
# archive_traces
# ---------------------------------------------------------------------------


def test_archive_delegates_to_rest_store():
    store = _make_rest_store(archive_traces=mock.MagicMock(return_value=7))
    result = archive_traces(store, older_than=timedelta(days=30))
    assert result == 7
    store.archive_traces.assert_called_once()


def test_archive_returns_zero_when_no_candidates():
    store = _make_local_store(collect_archive_candidates=mock.MagicMock(return_value=[]))
    result = archive_traces(store, older_than=timedelta(days=30))
    assert result == 0


def test_archive_passes_cutoff_ms_to_store():
    with mock.patch(
        "mlflow.tracing.trace_repo.get_current_time_millis", return_value=1_000_000_000
    ):
        store = _make_local_store()
        archive_traces(store, older_than=timedelta(days=10))
        store.collect_archive_candidates.assert_called_once()
        call_kwargs = store.collect_archive_candidates.call_args[1]
        expected_cutoff = 1_000_000_000 - int(timedelta(days=10).total_seconds() * 1000)
        assert call_kwargs["cutoff_ms"] == expected_cutoff


def test_archive_candidates_and_writes_artifacts():
    archive_data = _make_archive_data("t1", 1)
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=[("t1", 1)]),
        read_trace_for_archive=mock.MagicMock(return_value=archive_data),
    )
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        result = archive_traces(store, trace_ids=["t1"])
    assert result == 1
    store.read_trace_for_archive.assert_called_once_with("t1", 1)
    store.mark_trace_archived.assert_called_once_with("t1", archive_data.artifact_uri)
    mock_get_repo.assert_called_once_with(archive_data.artifact_uri)
    mock_get_repo.return_value.upload_trace_data.assert_called_once()


def test_archive_skips_when_read_returns_none():
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=[("t1", 1)]),
        read_trace_for_archive=mock.MagicMock(return_value=None),
    )
    result = archive_traces(store, trace_ids=["t1"])
    assert result == 0
    store.mark_trace_archived.assert_not_called()


def test_archive_skips_on_exception():
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=[("t1", 1), ("t2", 2)]),
        read_trace_for_archive=mock.MagicMock(
            side_effect=[RuntimeError("boom"), _make_archive_data("t2", 2)]
        ),
    )
    with mock.patch("mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"):
        result = archive_traces(store, trace_ids=["t1"])
    assert result == 1
    store.mark_trace_archived.assert_called_once()


def test_archive_batch_processing():
    candidates = [(f"t{i}", i) for i in range(5)]
    store = _make_local_store(
        collect_archive_candidates=mock.MagicMock(return_value=candidates),
        read_trace_for_archive=mock.MagicMock(
            side_effect=[_make_archive_data(f"t{i}", i) for i in range(5)]
        ),
    )
    with mock.patch("mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"):
        result = archive_traces(store, trace_ids=["t0"], batch_size=2)
    assert result == 5
    assert store.mark_trace_archived.call_count == 5


# ---------------------------------------------------------------------------
# load_archived_spans
# ---------------------------------------------------------------------------


def test_load_archived_raises_when_no_uri():
    store = _make_local_store(
        get_trace_repository_artifact_uri=mock.MagicMock(return_value=None),
    )
    trace_info = mock.MagicMock()
    trace_info.trace_id = "t1"
    with pytest.raises(MlflowTracingException, match="no artifact URI"):
        load_archived_spans(store, trace_info)


def test_load_archived_loads_from_artifact_repo():
    from mlflow.tracing.constant import SpansLocation

    spans = [mock.MagicMock()]
    store = _make_local_store(
        get_trace_repository_artifact_uri=mock.MagicMock(return_value="file:///repo/t1"),
    )
    trace_info = mock.MagicMock()
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        mock_get_repo.return_value.download_trace_data.return_value = spans
        result = load_archived_spans(store, trace_info)
    assert result == spans
    mock_get_repo.assert_called_once_with("file:///repo/t1")
    mock_get_repo.return_value.download_trace_data.assert_called_once_with(
        spans_location=SpansLocation.ARCHIVE_REPO
    )


# ---------------------------------------------------------------------------
# delete_traces
# ---------------------------------------------------------------------------


def test_delete_delegates_to_rest_store():
    store = _make_rest_store(delete_traces=mock.MagicMock(return_value=3))
    result = delete_traces(store, experiment_id="1", trace_ids=["t1"])
    assert result == 3
    store.delete_traces.assert_called_once()


def test_delete_cleans_up_artifacts_then_deletes_rows():
    store = _make_local_store(
        find_archived_trace_uris=mock.MagicMock(return_value={"t1": "file:///repo/1/traces/t1"}),
        delete_trace_rows=mock.MagicMock(return_value=2),
    )
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        result = delete_traces(store, experiment_id="1", trace_ids=["t1", "t2"])
    assert result == 2
    mock_get_repo.assert_called_once_with("file:///repo/1/traces/t1")
    mock_get_repo.return_value.delete_artifacts.assert_called_once()
    store.delete_trace_rows.assert_called_once()


def test_delete_best_effort_artifact_cleanup():
    store = _make_local_store(
        find_archived_trace_uris=mock.MagicMock(return_value={"t1": "file:///repo/1/traces/t1"}),
        delete_trace_rows=mock.MagicMock(return_value=1),
    )
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository"
    ) as mock_get_repo:
        mock_get_repo.return_value.delete_artifacts.side_effect = RuntimeError("disk error")
        result = delete_traces(store, experiment_id="1", trace_ids=["t1"])
    assert result == 1
    store.delete_trace_rows.assert_called_once()
