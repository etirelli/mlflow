import importlib
import json
import time
from unittest import mock

import pytest
from click.testing import CliRunner
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource as _OTelResource
from opentelemetry.sdk.trace import ReadableSpan as OTelReadableSpan

from mlflow.entities import trace_location
from mlflow.entities.span import Span, create_mlflow_span
from mlflow.entities.trace_info import TraceInfo
from mlflow.entities.trace_state import TraceState
from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import (
    MLFLOW_TRACE_ARCHIVAL_LOCATION,
    MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB,
    MLFLOW_TRACE_ARCHIVAL_OLDER_THAN,
    MLFLOW_TRACE_SPANS_STORAGE,
)
from mlflow.exceptions import MlflowException, MlflowTracingException
from mlflow.store.tracking.abstract_store import AbstractStore
from mlflow.store.tracking.dbmodels.models import SqlSpan, SqlTraceInfo, SqlTraceTag
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.tracing.constant import SpansLocation, TraceTagKey
from mlflow.tracing.utils import TraceJSONEncoder
from mlflow.utils.mlflow_tags import MLFLOW_ARTIFACT_LOCATION

pytestmark = pytest.mark.notrackingurimock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_span_context(trace_id_num=12345, span_id_num=111):
    ctx = mock.Mock()
    ctx.trace_id = trace_id_num
    ctx.span_id = span_id_num
    ctx.is_remote = False
    ctx.trace_flags = trace_api.TraceFlags(1)
    ctx.trace_state = trace_api.TraceState()
    return ctx


def _make_span(
    trace_id,
    name="test_span",
    span_id=111,
    parent_id=None,
    start_ns=1_000_000_000,
    end_ns=2_000_000_000,
    span_type="LLM",
    trace_num=12345,
    attributes=None,
) -> Span:
    context = _mock_span_context(trace_num, span_id)
    parent_context = _mock_span_context(trace_num, parent_id) if parent_id else None
    attributes = attributes or {}
    otel_span = OTelReadableSpan(
        name=name,
        context=context,
        parent=parent_context,
        attributes={
            "mlflow.traceRequestId": json.dumps(trace_id),
            "mlflow.spanType": json.dumps(span_type, cls=TraceJSONEncoder),
            **{k: json.dumps(v, cls=TraceJSONEncoder) for k, v in attributes.items()},
        },
        start_time=start_ns,
        end_time=end_ns,
        status=trace_api.Status(trace_api.StatusCode.UNSET),
        resource=_OTelResource.get_empty(),
    )
    return create_mlflow_span(otel_span, trace_id, span_type)


def _create_trace(
    store: SqlAlchemyStore,
    trace_id: str,
    experiment_id,
    request_time=0,
    tags=None,
) -> TraceInfo:
    trace_info = TraceInfo(
        trace_id=trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(experiment_id),
        request_time=request_time,
        execution_duration=0,
        state=TraceState.OK,
        tags=tags or {},
        trace_metadata={},
    )
    return store.start_trace(trace_info)


def _create_trace_with_spans(
    store: SqlAlchemyStore,
    trace_id: str,
    experiment_id,
    request_time=0,
    num_spans=1,
    span_content_size_hint=100,
    tags=None,
):
    trace_info = _create_trace(store, trace_id, experiment_id, request_time, tags)
    # Span start/end times must be >= request_time (in nanoseconds) to avoid
    # log_spans overwriting the trace timestamp_ms via min(current, span_start).
    start_ns = request_time * 1_000_000  # convert ms to ns
    spans = []
    for i in range(num_spans):
        attrs = {"mlflow.spanInputs": "x" * span_content_size_hint}
        span = _make_span(
            trace_id,
            name=f"span_{i}",
            span_id=111 + i,
            parent_id=None if i == 0 else 111,
            start_ns=start_ns + i * 1_000_000,
            end_ns=start_ns + 1_000_000_000 + i * 1_000_000,
            span_type="LLM",
            trace_num=12345 + hash(trace_id) % 1000,
            attributes=attrs,
        )
        spans.append(span)
    store.log_spans(experiment_id, spans)
    return trace_info


# ===========================================================================
# 1. archive_traces - time-based archival
# ===========================================================================


def test_archive_traces_older_than_duration(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_time")
    now_ms = int(time.time() * 1000)
    old_ms = now_ms - 2 * 86_400_000  # 2 days ago

    _create_trace_with_spans(store, "tr-old-1", exp_id, request_time=old_ms)
    _create_trace_with_spans(store, "tr-old-2", exp_id, request_time=old_ms - 1000)
    _create_trace_with_spans(store, "tr-recent", exp_id, request_time=now_ms)

    archived = store.archive_traces(older_than="1d")

    assert archived == 2

    # Verify old traces have TRACES_REPO location
    for tid in ("tr-old-1", "tr-old-2"):
        info = store.get_trace_info(tid)
        assert info.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACES_REPO.value

    # Verify recent trace still has TRACKING_STORE location
    recent_info = store.get_trace_info("tr-recent")
    assert recent_info.tags.get(TraceTagKey.SPANS_LOCATION) != SpansLocation.TRACES_REPO.value


def test_archive_traces_no_candidates(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_none")
    now_ms = int(time.time() * 1000)
    _create_trace_with_spans(store, "tr-new", exp_id, request_time=now_ms)

    archived = store.archive_traces(older_than="1d")
    assert archived == 0


def test_archive_traces_idempotency(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_idempotent")
    old_ms = int(time.time() * 1000) - 2 * 86_400_000

    _create_trace_with_spans(store, "tr-idem", exp_id, request_time=old_ms)

    first_run = store.archive_traces(older_than="1d")
    assert first_run == 1

    second_run = store.archive_traces(older_than="1d")
    assert second_run == 0


# ===========================================================================
# 2. archive_traces - size-based archival
# ===========================================================================


def test_archive_to_fit_under_size_limit(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_size")
    now_ms = int(time.time() * 1000)

    # Create several traces with known content
    for i in range(5):
        _create_trace_with_spans(
            store,
            f"tr-size-{i}",
            exp_id,
            request_time=now_ms - (5 - i) * 1000,  # oldest first
            span_content_size_hint=200,
        )

    # Since archive_traces takes int MB, use 0 to force archiving nearly everything
    archived = store.archive_traces(max_db_size_mb=0)
    assert archived >= 1


def test_archive_size_no_action_when_under_limit(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_size_ok")
    now_ms = int(time.time() * 1000)
    _create_trace_with_spans(store, "tr-small", exp_id, request_time=now_ms)

    # 1000 MB is way more than what the test DB would have
    archived = store.archive_traces(max_db_size_mb=1000)
    assert archived == 0


# ===========================================================================
# 3. archive_traces - experiment filter
# ===========================================================================


def test_archive_scoped_to_experiment(store: SqlAlchemyStore):
    exp1 = store.create_experiment("archive_exp1")
    exp2 = store.create_experiment("archive_exp2")
    old_ms = int(time.time() * 1000) - 2 * 86_400_000

    _create_trace_with_spans(store, "tr-e1", exp1, request_time=old_ms)
    _create_trace_with_spans(store, "tr-e2", exp2, request_time=old_ms)

    archived = store.archive_traces(experiment_id=exp1, older_than="1d")
    assert archived == 1

    # exp1 trace archived
    info_e1 = store.get_trace_info("tr-e1")
    assert info_e1.tags.get(TraceTagKey.SPANS_LOCATION) == SpansLocation.TRACES_REPO.value

    # exp2 trace untouched
    info_e2 = store.get_trace_info("tr-e2")
    assert info_e2.tags.get(TraceTagKey.SPANS_LOCATION) != SpansLocation.TRACES_REPO.value


# ===========================================================================
# 4. archive_traces - duration parsing
# ===========================================================================


@pytest.mark.parametrize(
    "duration_str",
    [
        "90d",
        "24h",
        "7d12h",
        "2m4s",
        "1d2h3m4s",
    ],
)
def test_valid_durations_do_not_raise(store: SqlAlchemyStore, duration_str):
    store.create_experiment(f"dur_{duration_str}")
    # Should not raise - just returns 0 because no traces match
    result = store.archive_traces(older_than=duration_str)
    assert result == 0


@pytest.mark.parametrize(
    "duration_str",
    [
        "abc",
        "90",
        "d",
        "",
        "1.2.3d",
    ],
)
def test_invalid_durations_raise(store: SqlAlchemyStore, duration_str):
    store.create_experiment(f"dur_bad_{duration_str or 'empty'}")
    with pytest.raises(MlflowException, match="Could not parse any time information"):
        store.archive_traces(older_than=duration_str)


# ===========================================================================
# 5. _get_spans_with_trace_info - dispatch by location
# ===========================================================================


def test_tracking_store_location_returns_spans(store: SqlAlchemyStore):
    exp_id = store.create_experiment("get_spans_ts")
    _create_trace_with_spans(store, "tr-ts", exp_id)

    trace_info = store.get_trace_info("tr-ts")
    with store.ManagedSessionMaker() as session:
        sql_spans = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-ts").all()
        spans = store._get_spans_with_trace_info(trace_info, sql_spans)

    assert spans is not None
    assert len(spans) >= 1


def test_traces_repo_location_dispatches_to_load_from_repo(store: SqlAlchemyStore):
    exp_id = store.create_experiment("get_spans_repo")
    _create_trace_with_spans(store, "tr-repo", exp_id)

    # Manually set the location tag to TRACES_REPO
    with store.ManagedSessionMaker() as session:
        session.merge(
            SqlTraceTag(
                request_id="tr-repo",
                key=TraceTagKey.SPANS_LOCATION,
                value=SpansLocation.TRACES_REPO.value,
            )
        )

    trace_info = store.get_trace_info("tr-repo")

    with mock.patch.object(
        store,
        "_load_spans_from_traces_repo",
        return_value=[],
    ) as mock_load:
        with store.ManagedSessionMaker() as session:
            sql_spans = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-repo").all()
            result = store._get_spans_with_trace_info(trace_info, sql_spans)
        mock_load.assert_called_once_with(trace_info)

    assert result == []


def test_artifact_repo_location_raises(store: SqlAlchemyStore):
    exp_id = store.create_experiment("get_spans_art")
    _create_trace_with_spans(store, "tr-art", exp_id)

    # Manually set the location tag to ARTIFACT_REPO
    with store.ManagedSessionMaker() as session:
        session.merge(
            SqlTraceTag(
                request_id="tr-art",
                key=TraceTagKey.SPANS_LOCATION,
                value=SpansLocation.ARTIFACT_REPO.value,
            )
        )

    trace_info = store.get_trace_info("tr-art")

    with store.ManagedSessionMaker() as session:
        sql_spans = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-art").all()
        with pytest.raises(MlflowTracingException, match="Trace data not stored"):
            store._get_spans_with_trace_info(trace_info, sql_spans)


# ===========================================================================
# 6. _load_spans_from_traces_repo
# ===========================================================================


def test_load_spans_from_traces_repo_happy_path(store: SqlAlchemyStore):
    exp_id = store.create_experiment("load_repo_happy")
    _create_trace_with_spans(store, "tr-load", exp_id, num_spans=2)

    # Archive the trace first to create the protobuf file
    old_ms = 0  # Very old
    with store.ManagedSessionMaker() as session:
        session.query(SqlTraceInfo).filter(
            SqlTraceInfo.request_id == "tr-load"
        ).update({"timestamp_ms": old_ms})

    archived = store.archive_traces(older_than="1s")
    assert archived == 1

    # Now load from repo
    trace_info = store.get_trace_info("tr-load")
    spans = store._load_spans_from_traces_repo(trace_info)

    assert len(spans) == 2
    # Root span (no parent) should come first
    assert spans[0].parent_id is None


def test_load_spans_from_traces_repo_missing_artifact_location(store: SqlAlchemyStore):
    exp_id = store.create_experiment("load_repo_noart")
    _create_trace_with_spans(store, "tr-noart", exp_id)

    # Remove the artifact location tag
    with store.ManagedSessionMaker() as session:
        session.query(SqlTraceTag).filter(
            SqlTraceTag.request_id == "tr-noart",
            SqlTraceTag.key == MLFLOW_ARTIFACT_LOCATION,
        ).delete()

    trace_info = store.get_trace_info("tr-noart")
    # Manually set traces_repo location to trigger the code path
    trace_info_dict = trace_info.tags.copy()
    trace_info_dict[TraceTagKey.SPANS_LOCATION] = SpansLocation.TRACES_REPO.value

    # Create a modified trace_info with the updated tags
    modified_info = TraceInfo(
        trace_id=trace_info.trace_id,
        trace_location=trace_location.TraceLocation.from_experiment_id(exp_id),
        request_time=trace_info.request_time,
        execution_duration=trace_info.execution_duration,
        state=trace_info.state,
        tags=trace_info_dict,
        trace_metadata=trace_info.trace_metadata,
    )

    with pytest.raises(MlflowTracingException, match="no artifact location"):
        store._load_spans_from_traces_repo(modified_info)


# ===========================================================================
# 7. content_size population
# ===========================================================================


def test_content_size_set_on_span_logging(store: SqlAlchemyStore):
    exp_id = store.create_experiment("content_size")
    _create_trace_with_spans(store, "tr-cs", exp_id, num_spans=1)

    with store.ManagedSessionMaker() as session:
        sql_span = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-cs").first()
        assert sql_span is not None
        assert sql_span.content_size > 0
        # Verify content_size matches actual content byte length
        expected_size = len(sql_span.content.encode("utf-8"))
        assert sql_span.content_size == expected_size


# ===========================================================================
# 8. CLI archive command
# ===========================================================================


def test_archive_command_exists():
    from mlflow.cli.traces import commands

    archive_cmd = next(
        (cmd for cmd in commands.commands.values() if cmd.name == "archive"), None
    )
    assert archive_cmd is not None
    param_names = [p.name for p in archive_cmd.params]
    assert "older_than" in param_names
    assert "max_db_size" in param_names
    assert "experiment_id" in param_names
    assert "backend_store_uri" in param_names


def test_archive_requires_older_than_or_max_db_size():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    result = runner.invoke(commands, ["archive"])
    assert result.exit_code != 0
    assert "Either --older-than or --max-db-size must be specified" in result.output


def test_archive_cli_with_older_than():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.archive_traces.return_value = 5
    mock_store.supports_workspaces = False

    with mock.patch(
        "mlflow.tracking._get_store", return_value=mock_store
    ) as mock_gs:
        result = runner.invoke(commands, ["archive", "--older-than", "30d"])

        assert result.exit_code == 0
        assert "Archived 5 trace(s)" in result.output
        mock_store.archive_traces.assert_called_once_with(
            experiment_id=None,
            older_than="30d",
            max_db_size_mb=None,
        )
        mock_gs.assert_called_once()


def test_archive_cli_with_max_db_size():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.archive_traces.return_value = 3
    mock_store.supports_workspaces = False

    with mock.patch(
        "mlflow.tracking._get_store", return_value=mock_store
    ) as mock_gs:
        result = runner.invoke(commands, ["archive", "--max-db-size", "1024"])

        assert result.exit_code == 0
        assert "Archived 3 trace(s)" in result.output
        mock_store.archive_traces.assert_called_once_with(
            experiment_id=None,
            older_than=None,
            max_db_size_mb=1024,
        )
        mock_gs.assert_called_once()


def test_archive_cli_with_experiment_id():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.archive_traces.return_value = 2
    mock_store.supports_workspaces = False

    with mock.patch(
        "mlflow.tracking._get_store", return_value=mock_store
    ) as mock_gs:
        result = runner.invoke(
            commands,
            ["archive", "--older-than", "7d", "--experiment-id", "42"],
        )

        assert result.exit_code == 0
        mock_store.archive_traces.assert_called_once_with(
            experiment_id="42",
            older_than="7d",
            max_db_size_mb=None,
        )
        mock_gs.assert_called_once()


# ===========================================================================
# 9. Workspace entity - traces_destination
# ===========================================================================


def test_workspace_to_dict_includes_traces_destination():
    ws = Workspace(
        name="test-ws",
        description="Test",
        default_artifact_root="s3://bucket/root",
        traces_destination="s3://bucket/traces",
    )
    d = ws.to_dict()
    assert d["traces_destination"] == "s3://bucket/traces"


def test_workspace_to_dict_traces_destination_none():
    ws = Workspace(name="test-ws")
    d = ws.to_dict()
    assert d["traces_destination"] is None


def test_workspace_from_dict_includes_traces_destination():
    payload = {
        "name": "ws1",
        "description": "desc",
        "default_artifact_root": "/root",
        "traces_destination": "gs://bucket/traces",
    }
    ws = Workspace.from_dict(payload)
    assert ws.traces_destination == "gs://bucket/traces"


def test_workspace_from_dict_without_traces_destination():
    payload = {"name": "ws1"}
    ws = Workspace.from_dict(payload)
    assert ws.traces_destination is None


def test_workspace_traces_destination_roundtrip():
    ws = Workspace(
        name="ws-rt",
        description="roundtrip",
        traces_destination="file:///tmp/traces",
    )
    ws2 = Workspace.from_dict(ws.to_dict())
    assert ws2.traces_destination == ws.traces_destination
    assert ws2.name == ws.name


# ===========================================================================
# 10. Environment variables
# ===========================================================================


def test_trace_archival_location_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACE_ARCHIVAL_LOCATION", raising=False)
    assert MLFLOW_TRACE_ARCHIVAL_LOCATION.get() is None


def test_trace_archival_location_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_ARCHIVAL_LOCATION", "s3://bucket/traces")
    assert MLFLOW_TRACE_ARCHIVAL_LOCATION.get() == "s3://bucket/traces"


def test_trace_spans_storage_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACE_SPANS_STORAGE", raising=False)
    assert MLFLOW_TRACE_SPANS_STORAGE.get() == "database"


def test_trace_spans_storage_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_SPANS_STORAGE", "external")
    assert MLFLOW_TRACE_SPANS_STORAGE.get() == "external"


def test_trace_archival_older_than_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACE_ARCHIVAL_OLDER_THAN", raising=False)
    assert MLFLOW_TRACE_ARCHIVAL_OLDER_THAN.get() is None


def test_trace_archival_older_than_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_ARCHIVAL_OLDER_THAN", "30d")
    assert MLFLOW_TRACE_ARCHIVAL_OLDER_THAN.get() == "30d"


def test_trace_archival_max_db_size_mb_default(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB", raising=False)
    assert MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB.get() is None


def test_trace_archival_max_db_size_mb_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB", "2048")
    assert MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB.get() == 2048


# ===========================================================================
# 11. SpansLocation enum
# ===========================================================================


def test_traces_repo_value():
    assert SpansLocation.TRACES_REPO.value == "TRACES_REPO"


def test_spans_location_all_values():
    values = {e.value for e in SpansLocation}
    assert "TRACKING_STORE" in values
    assert "ARTIFACT_REPO" in values
    assert "TRACES_REPO" in values


# ===========================================================================
# 12. Migration file structure
# ===========================================================================


def test_migration_module_loads():
    mod = importlib.import_module(
        "mlflow.store.db_migrations.versions.f2a3b4c5d6e7_add_trace_archival_columns"
    )
    assert mod.revision == "f2a3b4c5d6e7"
    assert mod.down_revision == "e1f2a3b4c5d6"
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)


def test_sqlspan_has_content_size_column():
    columns = {c.name for c in SqlSpan.__table__.columns}
    assert "content_size" in columns


def test_content_size_column_properties():
    col = SqlSpan.__table__.columns["content_size"]
    assert not col.nullable
    assert str(col.server_default.arg) == "0"


# ===========================================================================
# 13. archive_traces - span content cleared after archival
# ===========================================================================


def test_span_content_cleared_after_archive(store: SqlAlchemyStore):
    exp_id = store.create_experiment("clear_content")
    old_ms = int(time.time() * 1000) - 2 * 86_400_000
    _create_trace_with_spans(store, "tr-clear", exp_id, request_time=old_ms)

    # Verify content exists before archival
    with store.ManagedSessionMaker() as session:
        span_before = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-clear").first()
        assert span_before.content != ""
        assert span_before.content_size > 0

    store.archive_traces(older_than="1d")

    # Verify content cleared and size reset
    with store.ManagedSessionMaker() as session:
        span_after = session.query(SqlSpan).filter(SqlSpan.trace_id == "tr-clear").first()
        assert span_after.content == ""
        assert span_after.content_size == 0


def test_trace_metadata_preserved_after_archive(store: SqlAlchemyStore):
    exp_id = store.create_experiment("preserve_meta")
    old_ms = int(time.time() * 1000) - 2 * 86_400_000
    _create_trace_with_spans(store, "tr-meta", exp_id, request_time=old_ms)

    info_before = store.get_trace_info("tr-meta")
    trace_id = info_before.trace_id

    store.archive_traces(older_than="1d")

    info_after = store.get_trace_info(trace_id)
    # Trace info should still be queryable
    assert info_after.trace_id == trace_id
    assert info_after.state == TraceState.OK
    assert info_after.experiment_id == exp_id


# ===========================================================================
# 14. AbstractStore.archive_traces raises NotImplemented
# ===========================================================================


def test_abstract_store_archive_traces_raises():
    from mlflow.exceptions import MlflowNotImplementedException

    store = AbstractStore.__new__(AbstractStore)
    with pytest.raises(MlflowNotImplementedException):  # noqa: PT011
        store.archive_traces()


# ===========================================================================
# 15. Workspace proto roundtrip (C3 fix)
# ===========================================================================


def test_workspace_to_proto_includes_traces_destination():
    ws = Workspace(
        name="ws-proto",
        description="Test",
        default_artifact_root="s3://bucket/root",
        traces_destination="s3://bucket/traces",
    )
    proto = ws.to_proto()
    assert proto.traces_destination == "s3://bucket/traces"
    assert proto.HasField("traces_destination")


def test_workspace_from_proto_includes_traces_destination():
    from mlflow.protos.service_pb2 import Workspace as ProtoWorkspace

    proto = ProtoWorkspace()
    proto.name = "ws-proto"
    proto.traces_destination = "gs://bucket/traces"
    ws = Workspace.from_proto(proto)
    assert ws.traces_destination == "gs://bucket/traces"


def test_workspace_proto_roundtrip_traces_destination():
    ws = Workspace(
        name="ws-rt",
        description="roundtrip",
        traces_destination="file:///tmp/traces",
    )
    ws2 = Workspace.from_proto(ws.to_proto())
    assert ws2.traces_destination == ws.traces_destination
    assert ws2.name == ws.name


def test_workspace_proto_no_traces_destination():
    ws = Workspace(name="ws-none")
    proto = ws.to_proto()
    assert not proto.HasField("traces_destination")
    ws2 = Workspace.from_proto(proto)
    assert ws2.traces_destination is None


# ===========================================================================
# 16. Per-trace error handling (I3 fix)
# ===========================================================================


def test_archive_continues_on_single_trace_failure(store: SqlAlchemyStore):
    exp_id = store.create_experiment("archive_error_handling")
    old_ms = int(time.time() * 1000) - 2 * 86_400_000

    # tr-older is processed first (oldest-first ordering), tr-newer second
    _create_trace_with_spans(store, "tr-newer", exp_id, request_time=old_ms)
    _create_trace_with_spans(store, "tr-older", exp_id, request_time=old_ms - 1000)

    # Make the artifact upload fail for the first trace (tr-older) but succeed for tr-newer
    with mock.patch(
        "mlflow.store.artifact.artifact_repository_registry.get_artifact_repository",
    ) as mock_repo:
        error_repo = mock.MagicMock()
        error_repo.log_artifact.side_effect = Exception("Upload failed")

        ok_repo = mock.MagicMock()

        # First call fails (tr-older), second succeeds (tr-newer)
        mock_repo.side_effect = [error_repo, ok_repo]

        archived = store.archive_traces(older_than="1d")

    # One trace should have been archived successfully despite the other failing
    assert archived == 1


# ===========================================================================
# 17. CLI workspace options exist (I1 fix)
# ===========================================================================


def test_archive_command_has_workspace_options():
    from mlflow.cli.traces import commands

    archive_cmd = next(
        (cmd for cmd in commands.commands.values() if cmd.name == "archive"), None
    )
    assert archive_cmd is not None
    param_names = [p.name for p in archive_cmd.params]
    assert "workspace" in param_names
    assert "all_workspaces" in param_names


def test_archive_cli_workspace_and_all_workspaces_conflict():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.supports_workspaces = False

    with mock.patch(
        "mlflow.tracking._get_store", return_value=mock_store
    ):
        result = runner.invoke(
            commands,
            ["archive", "--older-than", "30d", "--workspace", "ws1", "--all-workspaces"],
        )
        assert result.exit_code != 0
        assert "Cannot use --workspace and --all-workspaces together" in result.output


# ===========================================================================
# 18. CLI env var wiring (S1 fix)
# ===========================================================================


def test_archive_cli_older_than_from_envvar():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.archive_traces.return_value = 2
    mock_store.supports_workspaces = False

    with mock.patch("mlflow.tracking._get_store", return_value=mock_store):
        result = runner.invoke(
            commands,
            ["archive"],
            env={"MLFLOW_TRACE_ARCHIVAL_OLDER_THAN": "45d"},
        )
        assert result.exit_code == 0
        mock_store.archive_traces.assert_called_once_with(
            experiment_id=None,
            older_than="45d",
            max_db_size_mb=None,
        )


def test_archive_cli_max_db_size_from_envvar():
    from mlflow.cli.traces import commands

    runner = CliRunner(catch_exceptions=False)
    mock_store = mock.MagicMock()
    mock_store.archive_traces.return_value = 1
    mock_store.supports_workspaces = False

    with mock.patch("mlflow.tracking._get_store", return_value=mock_store):
        result = runner.invoke(
            commands,
            ["archive"],
            env={"MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB": "512"},
        )
        assert result.exit_code == 0
        mock_store.archive_traces.assert_called_once_with(
            experiment_id=None,
            older_than=None,
            max_db_size_mb=512,
        )
