# Installing ml4paleo v2 on one machine

Everything runs in Docker Compose from `deploy/compose/`: Caddy (HTTPS, the
only service with published ports), the API (which also serves the web app),
the housekeeper, Postgres, SeaweedFS (object storage), and the job workers.

## What you need

- A Linux machine with Docker and the Compose plugin (2.24 or newer). 16 GB
  of memory and a few CPU cores go a long way; disk for your scans (about the
  size of the uploads, plus the same again for predictions and exports).
- A domain name pointing at the machine, with ports 80 and 443 open, so Caddy
  can get a certificate from Let's Encrypt. To try it out, use `localhost`
  instead (Caddy then uses its own local certificate authority).
- Optionally, an NVIDIA GPU with the NVIDIA Container Toolkit, for the GPU
  worker.
- Optionally, an SMTP account for email (people confirming their addresses,
  password resets, invites, notices about requests for more storage).
  Without one, admins confirm addresses on the admin page.

## First start

```sh
git clone -b v2 https://github.com/j6k4m8/ml4paleo.git
cd ml4paleo/deploy/compose
./setup.sh ml4paleo.example.org     # or: ./setup.sh localhost
docker compose up -d --build
```

`setup.sh` writes random secrets into `secrets/` and a starter `.env`. It never
overwrites existing files, so it is safe to run again. If it finds an NVIDIA
GPU that Docker can use, it turns on the GPU worker (`COMPOSE_PROFILES=gpu`).

Then open `https://ml4paleo.example.org` and sign in as `admin` with the
password in `secrets/initial_admin_password`. You'll be asked to choose a new
password and to set up two-factor sign-in with an authenticator app; admins
can't skip either. To get requests for more storage by email, give the admin
account an address:

```sh
docker compose exec api ml4paleo-server set-email admin you@example.org
```

To check that workers pick up jobs:

```sh
docker compose exec api ml4paleo-server check-workers
```

## Settings

`.env` holds the settings, and the server reads every `M4P_` setting in it.
These are set up for you:

| Setting | What it does |
|---|---|
| `M4P_DOMAIN` | The domain people use. |
| `M4P_SMTP__HOST`, `__PORT`, `__USERNAME`, `__FROM_ADDRESS` | Outgoing email; leave the host empty to turn email off. The password goes in `secrets/smtp_password`. For implicit TLS (usually port 465), add `M4P_SMTP__SECURITY=tls`. |
| `COMPOSE_PROFILES` | `gpu` also runs the GPU worker. |
| `M4P_CPU_WORKER_SLOTS`, `M4P_CPU_WORKER_MEMORY` | How many jobs the CPU worker runs at once, and its memory limit. Each job sizes itself to its share of the memory. |
| `M4P_GPU_WORKER_MEMORY` | The GPU worker's memory limit. |
| `M4P_API_WORKERS` | API processes (default 4). |

Others worth knowing (nested settings use `__`; see
`server/ml4paleo_server/settings.py` for all of them):

| Setting | Default | What it does |
|---|---|---|
| `M4P_QUOTA__STORAGE_GB` | 10 | Storage per account (projects count against their owner). |
| `M4P_QUOTA__TRAINED_MODELS` | 20 | Trained models each account can keep. |
| `M4P_AUTH__SIGNUP_MODE` | open | Who can sign up at first: `open`, or `invite` (admins can change it later). |
| `M4P_AUTH__REQUIRE_EMAIL` | true | Whether sign-up asks for an email address, with starter limits until it's confirmed (admins can change it later). |
| `M4P_UNCONFIRMED_QUOTA__STORAGE_GB`, `M4P_UNCONFIRMED_QUOTA__TRAINED_MODELS` | 1, 1 | The starter limits. |
| `M4P_AUTH__SESSION_IDLE_DAYS`, `M4P_AUTH__SESSION_MAX_DAYS` | 7, 30 | When sessions end. |
| `M4P_LABEL_CACHE_MB` | 64 | Memory each API process keeps for the zoomed-out views of labels, which it works out as viewers ask for them. |

To lift a limit entirely, set the whole group as JSON, for example
`M4P_QUOTA={"storage_gb": null, "trained_models": 20}`. After changing `.env`,
run `docker compose up -d` again.

## Storage

Scans, labels, models, and results live in SeaweedFS (the
`ml4paleo_seaweedfs-data` volume). Browsers upload straight to it through
Caddy, with signed URLs for one file part each; nothing else of SeaweedFS is
reachable from outside.

To use a cloud bucket instead (AWS S3, Cloudflare R2, or Google Cloud Storage
through its S3 API), put the storage settings for `migrate`, `api`, and
`housekeeper` in `compose.override.yml` next to `compose.yml` (Compose reads
it on its own):

```yaml
x-storage: &storage
  M4P_STORAGE__URL: s3://your-bucket
  M4P_STORAGE__ENDPOINT: https://s3.us-east-1.amazonaws.com
  M4P_STORAGE__PUBLIC_ENDPOINT: https://s3.us-east-1.amazonaws.com
  M4P_STORAGE__REGION: us-east-1
services:
  migrate: { environment: *storage }
  api: { environment: *storage }
  housekeeper: { environment: *storage }
```

Put the bucket's access keys in `secrets/s3_access_key_id` and
`secrets/s3_secret_access_key`. Browsers then upload to the bucket's own
site, so give the bucket a CORS rule allowing `PUT` from your site's address
(`https://ml4paleo.example.org`); the app's security policy allows that
address on its own. Workers on this machine still reach storage through the
site.

## Backups

The `backup` service dumps the database once a day (counting from when it
starts) into `deploy/compose/backups`, and keeps two weeks of dumps. Copy off
the machine, regularly:

- `deploy/compose/backups/`
- `deploy/compose/secrets/` (the database password, and the key that protects
  two-factor secrets and queued email: a database restored without it can't
  check anyone's two-factor codes)
- the `ml4paleo_seaweedfs-data` volume (or keep your bucket's own backups)

To restore onto a new machine: copy `secrets/` (and `.env`) into
`deploy/compose/` first, then run `./setup.sh` and `docker compose up -d
--build` as for a first start, and restore the SeaweedFS volume. Then load the
newest dump:

```sh
docker compose stop api housekeeper backup worker-cpu
docker compose exec postgres dropdb -U ml4paleo --force ml4paleo
docker compose exec postgres createdb -U ml4paleo ml4paleo
docker compose exec -T postgres pg_restore -U ml4paleo -d ml4paleo --exit-on-error \
  < backups/ml4paleo-YYYYMMDD-HHMMSS.dump
docker compose up -d
```

`up` runs `migrate` again, which brings an older dump's database up to date
and registers this machine's local worker token.

## Upgrading

```sh
git pull
docker compose up -d --build
```

The `migrate` service upgrades the database before the API starts.

## Importing jobs from ml4paleo v1

v2 replaces the v1 app on the same site, and v1's job links (`/job/<id>`)
keep working: signed in, someone visiting one can bring that job over into a
new project (its scan, its placeable annotation samples as labels, and its
last finished segmentation as the prediction). The Import page also lists
the jobs a browser opened in v1, which can include jobs other people shared,
to import one at a time; it works only when v2 is served from exactly the
address v1 was (scheme, host, and port), since browsers keep that list per
site. The first person to import a job gets it; admins can give a job claimed
by the wrong person to its owner.

Mount v1's volume folder (the one with `jobs.json`) read-only with the v1
override, which also starts a worker that can read it. Put both lines in
`.env`, so a plain `docker compose up -d` keeps using it:

```sh
M4P_V1_VOLUME=/home/ubuntu/ml4paleo-webapp-volume
COMPOSE_FILE=compose.yml:compose.v1.yml
```

The API sees `jobs.json` as it was when the API started, since v1 saves it by
replacing the file. If v1 is still running, run `docker compose restart api`
to pick up the jobs made since.
