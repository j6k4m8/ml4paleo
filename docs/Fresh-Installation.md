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
