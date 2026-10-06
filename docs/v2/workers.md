# Workers

Workers run everything heavy: reading uploads, building image pyramids,
training models, predicting, composing final segmentations, meshing, and
exports. They pull jobs from the site over HTTPS (they never need inbound
connections), hold a lease on each job while they work, and report back. If a
worker stops answering, its jobs go back in the queue; a job gets three tries.

## On the site's machine

`deploy/compose` runs `worker-cpu`, and `worker-gpu` too when the GPU profile
is on. They reach only the worker API, through Caddy on a private network,
and read and write storage through the site, never directly.

- `M4P_CPU_WORKER_SLOTS`: jobs at once (default 1). Each job gets an equal
  share of the worker's memory and CPUs and sizes its work to fit.
- `M4P_CPU_WORKER_MEMORY`: the worker's memory limit (default `4g`). More
  memory lets jobs work in bigger pieces; very large scans want 8–16 GB.

## On another machine

To add, say, a lab workstation with a GPU:

1. In **Administration** → **Workers**, name the worker and make its token.
   Copy the token (it's shown once) into a file on that machine, readable
   only by the user who runs the worker.
2. On that machine, with Docker:

   ```sh
   docker build -f deploy/docker/worker.Dockerfile -t ml4paleo-worker .
   docker run -d --restart unless-stopped \
     -v /path/to/token:/run/secrets/worker_token:ro \
     ml4paleo-worker ml4paleo-worker \
       --server https://ml4paleo.example.org \
       --token-file /run/secrets/worker_token
   ```

   or, with Python 3.12 and uv, from a checkout:

   ```sh
   uv sync --package ml4paleo-worker
   uv run ml4paleo-worker --server https://ml4paleo.example.org --token-file ~/worker_token
   ```

The worker finds a GPU by itself (and then also takes GPU jobs). Options, each
also an environment variable:

| Option | Variable | What it does |
|---|---|---|
| `--slots N` | `M4PW_SLOTS` | Jobs at once. |
| `--label NAME` | `M4PW_LABELS` | Labels some jobs ask for (comma-separated in the variable). |
| `--v1-volume PATH` | `M4PW_V1_VOLUME` | The v1 app's volume folder, to import v1 jobs (adds the `v1-volume` label). |

To stop a worker, stop its process: it gives its running jobs back first. To
take its access away, revoke it in **Administration** → **Workers**.
