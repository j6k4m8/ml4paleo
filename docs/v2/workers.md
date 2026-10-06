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

To add, say, a lab workstation with lots of memory or a GPU:

1. In **Administration** → **Workers**, name the worker and make its token.
   It's shown once.
2. On that machine, with Docker and a copy of the v2 code:

   ```sh
   git clone -b v2 https://github.com/j6k4m8/ml4paleo.git && cd ml4paleo
   docker build -f deploy/docker/worker.Dockerfile -t ml4paleo-worker .
   # The worker runs as user 10002 in its container, so let it read the token.
   sudo install -m 0400 -o 10002 /dev/stdin /etc/ml4paleo-worker-token <<< "PASTE-THE-TOKEN"
   docker run -d --restart unless-stopped --init --stop-timeout 60 \
     --gpus all \
     -v /etc/ml4paleo-worker-token:/run/secrets/worker_token:ro \
     ml4paleo-worker ml4paleo-worker \
       --server https://ml4paleo.example.org \
       --token-file /run/secrets/worker_token
   ```

   Leave out `--gpus all` on a machine without an NVIDIA GPU (or without the
   NVIDIA Container Toolkit). `--stop-timeout 60` gives a stopping worker time
   to hand its jobs back.

   Or, with Python 3.12 and uv, from the checkout:

   ```sh
   uv run --package ml4paleo-worker ml4paleo-worker \
     --server https://ml4paleo.example.org --token-file ~/worker_token
   ```

A worker on another machine reads and writes storage through the site, so it
needs nothing but HTTPS to it. It finds an NVIDIA GPU by itself and says so to
the site, for jobs that want one. Options, each also an environment variable:

| Option | Variable | What it does |
|---|---|---|
| `--server URL` | `M4PW_SERVER` | The site's address. |
| `--token-file PATH` | `M4PW_TOKEN_FILE` | The file holding the worker's token. |
| `--slots N` | `M4PW_SLOTS` | Jobs at once. |
| `--label NAME` | `M4PW_LABELS` | Labels some jobs ask for (comma-separated in the variable). |
| `--v1-volume PATH` | `M4PW_V1_VOLUME` | The v1 app's volume folder, to import v1 jobs (adds the `v1-volume` label). |
| `--allow-http` | `M4PW_ALLOW_HTTP=true` | Allow a plain `http://` server address, on a private network only. |

To stop a worker, stop its process: it gives its running jobs back first. To
take its access away, revoke it in **Administration** → **Workers**.
