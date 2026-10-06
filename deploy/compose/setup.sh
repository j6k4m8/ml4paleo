#!/bin/sh
# Generate secrets and a starter .env for the single-machine compose deploy.
#
#   ./setup.sh ml4paleo.example.org   # a real domain, with HTTPS from Let's Encrypt
#   ./setup.sh localhost              # try it locally (HTTPS with a local CA)
#
# Existing files are never overwritten, so it is safe to run again.
set -eu

if [ $# -ne 1 ]; then
    echo "usage: $0 DOMAIN   (for example: $0 ml4paleo.example.org, or $0 localhost)" >&2
    exit 2
fi
domain=$1

cd "$(dirname "$0")"
umask 077
mkdir -p secrets backups

random() {
    head -c 256 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c "$1"
}

write_secret() {
    if [ ! -f "secrets/$1" ]; then
        printf '%s' "$2" > "secrets/$1"
        echo "Created secrets/$1"
    fi
}

write_secret secret_key "$(random 64)"
write_secret postgres_password "$(random 32)"
write_secret s3_access_key_id "m4p$(random 17)"
write_secret s3_secret_access_key "$(random 40)"
write_secret database_url \
    "postgresql+psycopg://ml4paleo:$(cat secrets/postgres_password)@postgres:5432/ml4paleo"
write_secret seaweedfs_s3.json "$(cat <<JSON
{
  "identities": [
    {
      "name": "ml4paleo",
      "credentials": [
        {
          "accessKey": "$(cat secrets/s3_access_key_id)",
          "secretKey": "$(cat secrets/s3_secret_access_key)"
        }
      ],
      "actions": ["Read", "List", "Tagging", "Write"]
    }
  ]
}
JSON
)"

# Fill in to send email with a password (leave empty otherwise).
write_secret smtp_password ""
# The first admin account's password. It must be changed at first sign-in.
write_secret initial_admin_password "$(random 24)"
# The token the local workers (worker-cpu, worker-gpu) share.
write_secret worker_token "m4pw_$(random 43)"

# Containers run as their own users, and compose mounts these files as-is, so
# they must be readable. The secrets directory itself stays private (0700).
chmod 0644 secrets/*

# Run the GPU worker too if this machine has an NVIDIA GPU that Docker can use
# (that needs the NVIDIA Container Toolkit).
profiles=
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
        profiles=gpu
    else
        echo "Found an NVIDIA GPU, but Docker can't use it. Install the NVIDIA" \
            "Container Toolkit, then set COMPOSE_PROFILES=gpu in .env." >&2
    fi
fi
if [ -n "$profiles" ]; then
    if [ ! -f .env ]; then
        echo "Found an NVIDIA GPU; the GPU worker will run too."
    elif ! grep -q '^COMPOSE_PROFILES=.*gpu' .env; then
        echo "Found an NVIDIA GPU. To run the GPU worker, set COMPOSE_PROFILES=gpu in .env."
    fi
fi

if [ ! -f .env ]; then
    cat > .env <<ENV
# The domain people use to reach ml4paleo. Caddy gets an HTTPS certificate for
# it automatically ("localhost" uses Caddy's own local certificate authority).
M4P_DOMAIN=$domain
# Optional outgoing email (confirming addresses, password resets). Leave
# M4P_SMTP__HOST empty to turn email off. Put the password in
# secrets/smtp_password.
M4P_SMTP__HOST=
M4P_SMTP__PORT=587
M4P_SMTP__USERNAME=
M4P_SMTP__FROM_ADDRESS=ml4paleo <no-reply@$domain>
# "gpu" also runs worker-gpu (set automatically when an NVIDIA GPU was found).
COMPOSE_PROFILES=$profiles
# How many jobs the CPU worker runs at once, and the memory it may use.
M4P_CPU_WORKER_SLOTS=1
M4P_CPU_WORKER_MEMORY=4g
M4P_GPU_WORKER_MEMORY=16g
ENV
    echo "Created .env for https://$domain"
fi

echo "Sign in at https://$domain as 'admin' with the password in $(pwd)/secrets/initial_admin_password"
