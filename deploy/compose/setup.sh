#!/bin/sh
# Generate secrets and a starter .env for the single-machine compose deploy.
# Existing files are never overwritten, so it is safe to run again.
set -eu

cd "$(dirname "$0")"
umask 077
mkdir -p secrets

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
      "actions": ["Admin", "Read", "List", "Tagging", "Write"]
    }
  ]
}
JSON
)"

# Containers run as their own users, and compose mounts these files as-is, so
# they must be readable. The secrets directory itself stays private (0700).
chmod 0644 secrets/*

if [ ! -f .env ]; then
    cat > .env <<ENV
# Your domain, for automatic HTTPS (for example ml4paleo.example.org).
# ":80" serves plain HTTP for local use.
M4P_SITE_ADDRESS=:80
# The URL people use to reach the app.
M4P_PUBLIC_URL=http://localhost
ENV
    echo "Created .env"
fi
