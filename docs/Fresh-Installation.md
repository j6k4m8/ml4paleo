# Fresh Installation on Ubuntu 24LTS

## Install docker

```
sudo snap install docker
```

Note that you also need `docker-compose`; depending on your version, you may need to install this separately.

_Also_ keep in mind that you _may_ need to add the current user to the docker group,

```
sudo usermod -aG docker $USER
```

...which requires logging out and back in to take effect.

## Clone the repo

```
git clone https://github.com/j6k4m8/ml4paleo
```

## Build the compose containers

```
docker compose up -d --build
```

## Create the nginx configuration

```
sudo apt install nginx
```

In `/etc/nginx/sites-available/default`, replace all contents with the following:

```
server {
    listen 80;
    server_name ml4paleo.example.org;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    client_max_body_size 512M;
}
```

## Get an SSL certificate from certbot

This is only required if you want to serve over HTTPS; you need to serve over HTTPS to use the included neuroglancer visualization links.

```
sudo snap install --classic certbot
```

Then create the cert by following the prompts:

```
sudo ln -s /snap/bin/certbot /usr/bin/certbot
```
