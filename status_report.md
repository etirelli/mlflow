## Phase 6: Status Report

### Implementation Summary

The trace archival feature (ticket-20574) has partial implementation across 10 files + 1 migration + 1 new untracked file. The core archival workflow (archive DB spans to protobuf, retrieve from trace repo) is functional. **47 tests were written and all pass.**

---

### Files Changed

| File                                                     | Change                                                                                                                        |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `mlflow/cli/__init__.py`                                 | `--trace-archival-location` server option                                                                                     |
| `mlflow/cli/traces.py`                                   | `mlflow traces archive` CLI subcommand                                                                                        |
| `mlflow/entities/workspace.py`                           | `traces_destination` field on Workspace entity                                                                                |
| `mlflow/environment_variables.py`                        | 4 new env vars (archival location, spans storage, retention defaults)                                                         |
| `mlflow/store/tracking/abstract_store.py`                | `archive_traces()` abstract method                                                                                            |
| `mlflow/store/tracking/dbmodels/models.py`               | `content_size` column on SqlSpan                                                                                              |
| `mlflow/store/tracking/sqlalchemy_store.py`              | Core: `archive_traces()`, `_archive_trace_batch()`, `_load_spans_from_traces_repo()`, modified `_get_spans_with_trace_info()` |
| `mlflow/store/workspace/dbmodels/models.py`              | `traces_destination` column on SqlWorkspace                                                                                   |
| `mlflow/store/workspace/sqlalchemy_store.py`             | Workspace create/update for `traces_destination`                                                                              |
| `mlflow/tracing/constant.py`                             | `TRACES_REPO` added to `SpansLocation` enum                                                                                   |
| `mlflow/store/db_migrations/versions/f2a3b4c5d6e7_...py` | Alembic migration                                                                                                             |
| `tests/store/tracking/test_trace_archival.py`            | **NEW** - 47 tests (all passing)                                                                                              |

---

### Critical Issues (3)

| #   | Issue                                                                                                                                                                                               | File:Line                       | Fix                                                                                  |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| C1  | **Empty `""` older_than archives ALL traces** — regex matches empty string, producing `timedelta(0)`, so cutoff = now and every trace qualifies                                                     | `sqlalchemy_store.py:4850-4863` | Add `if not time_params: raise MlflowException(...)` after parsing                   |
| C2  | **Single DB transaction for entire archival** — all batches run inside one `ManagedSessionMaker` context. Crash loses all progress; memory grows unboundedly                                        | `sqlalchemy_store.py:4868`      | Move `ManagedSessionMaker` inside the batch loop so each batch commits independently |
| C3  | **`Workspace.to_proto()`/`from_proto()` drops `traces_destination`** — the field is not serialized to/from protobuf, so REST API round-trips silently lose it. Proto definition not updated either. | `workspace.py:51-70`            | Add field to proto definition and both methods                                       |

### Important Issues (4)

| #   | Issue                                                                                                                                          | File:Line                                       | Fix                                                          |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| I1  | **No workspace-scoping** — design requires `--workspace`/`--all-workspaces`; size-based path bypasses `_trace_query()` (workspace-aware query) | `traces.py:815-893`, `sqlalchemy_store.py:4877` | Add workspace CLI options; use `_trace_query()` consistently |
| I2  | **Size-based `total_size` ignores `experiment_id` filter** — computes global total even when scoped to one experiment                          | `sqlalchemy_store.py:4872`                      | Filter `total_size` query by experiment_id when specified    |
| I3  | **No per-trace error handling** — one failed artifact upload aborts entire batch                                                               | `sqlalchemy_store.py:4965-5065`                 | Wrap per-trace logic in try/except, log warning, continue    |
| I4  | **Unbounded protobuf deserialization** — `f.read()` on archived `traces.pb` has no size limit; large/malicious file causes OOM                 | `sqlalchemy_store.py:4811-4815`                 | Add file size check before reading                           |

### Suggestions (4)

| #   | Issue                                                                                                                                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------ |
| S1  | Env var defaults (`MLFLOW_TRACE_ARCHIVAL_OLDER_THAN`, `MLFLOW_TRACE_ARCHIVAL_MAX_DB_SIZE_MB`) not wired to CLI options via `envvar=` |
| S2  | `MLFLOW_TRACE_SPANS_STORAGE` env var defined but unused (future "repository" mode)                                                   |
| S3  | No progress logging during long-running archival                                                                                     |
| S4  | Duration regex accepts malformed floats like `"1.2.3d"` which produce unhelpful `ValueError`                                         |

### Security Review Summary

- **No SQL injection** — all queries use SQLAlchemy ORM with parameterized filters
- **No path traversal** — trace IDs come from DB, `experiment_id` cast to `int()`, `append_to_uri_path` used correctly
- **Temp files safe** — `tempfile.TemporaryDirectory()` with context managers
- **No ReDoS** — regex is anchored and uses non-overlapping character classes
- **1 actionable finding**: Unbounded protobuf deserialization (covered in I4 above)

### Test Coverage (NEW)

**47 tests created**, all passing in 1.77s:

| Area                                 | Tests |
| ------------------------------------ | ----- |
| Time-based archival                  | 3     |
| Size-based archival                  | 2     |
| Experiment filter                    | 1     |
| Duration parsing (valid/invalid/bug) | 9     |
| Span location dispatch               | 3     |
| Load from traces repo                | 2     |
| content_size population              | 1     |
| CLI archive command                  | 5     |
| Workspace entity                     | 5     |
| Environment variables                | 8     |
| SpansLocation enum                   | 2     |
| Migration structure                  | 3     |
| Post-archival state                  | 2     |
| AbstractStore fallback               | 1     |

### What's Done Well

- OTLP protobuf format using existing `Span.to_otel_proto()`/`from_otel_proto()` is well-chosen
- Retrieval dispatch in `_get_spans_with_trace_info` is clean and transparent
- `content_size` computed as byte length (not character length) at write time
- Migration uses `batch_alter_table` for SQLite compatibility and conditionally checks for `workspaces` table
- Idempotent archival design — `TRACKING_STORE` traces selected, protobuf overwritten, then tag updated
