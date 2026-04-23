# Developing

This repo uses `uv` for local development. The source of truth is the root `pyproject.toml` plus `uv.lock`.

## Setup

From the repo root:

```bash
uv sync
mkdir -p webapp/volume
```

The base install comes from `[project.dependencies]` in `pyproject.toml`.

If you need the optional dependency groups defined in `pyproject.toml`, install them explicitly:

```bash
uv sync --group bossdb
uv sync --group dicom
uv sync --group bossdb --group dicom
```

## Important Path Note

Run the webapp commands with `--directory webapp`.

That matters for two reasons:

1. The webapp scripts use local imports like `job` and `config`.
2. The runtime data directory is relative to `webapp/`, e.g. `webapp/volume/jobs.json`.

If you run `uv run python webapp/main.py` from the repo root, imports and `volume/` paths can break.

## Frontend + Backend Hot Reload

The UI is server-rendered from Flask templates in `webapp/templates/`, and the backend routes live in `webapp/main.py`.

To run the local development server with Flask's debug reloader:

```bash
uv run --directory webapp python main.py
```

Then open <http://localhost:5000>.

Hot reload behavior:

-   Python changes in the Flask app trigger the debug reloader.
-   Template changes in `webapp/templates/` are picked up on the next browser refresh.
-   There is no separate frontend dev server to start.

## Background Workers

For the full upload -> convert -> segment -> mesh workflow, run the workers in separate terminals:

```bash
uv run --directory webapp python conversionrunner.py
uv run --directory webapp python segmentrunner.py
uv run --directory webapp python meshrunner.py
```

These workers are simple polling loops. They do not hot reload themselves, so restart a worker after editing its code.

## Typical Local Workflow

Terminal 1:

```bash
uv run --directory webapp python main.py
```

Terminal 2:

```bash
uv run --directory webapp python conversionrunner.py
```

Terminal 3:

```bash
uv run --directory webapp python segmentrunner.py
```

Terminal 4:

```bash
uv run --directory webapp python meshrunner.py
```

If you are only working on the UI or Flask routes, Terminal 1 is usually enough.

## Troubleshooting

-   `ModuleNotFoundError: No module named 'job'`
    Use `uv run --directory webapp ...` instead of running the script from the repo root.
-   Missing `volume/jobs.json` or related file errors
    Make sure `webapp/volume` exists.
-   Dependency changes not showing up
    Re-run `uv sync` after editing `pyproject.toml`.
