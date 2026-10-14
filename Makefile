# Docker build and run helpers.

IMAGE ?= streaming-vlms:latest
CONTAINER ?= streaming-vlms

.PHONY: build boot-attached

## Build the docker image.
build:
	docker build -f docker/Dockerfile --tag $(IMAGE) .

## Start an interactive GPU container from the built image.
boot-attached:
	docker run --gpus all --name $(CONTAINER) -it --rm $(IMAGE)
