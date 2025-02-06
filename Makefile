PROJECT ?= Cosmos
NAME ?= Cosmos
VERSION ?= 251

WORKSPACE ?= /workspace/$(PROJECT)
DOCKER_IMAGE ?= cosmos:latest

SAGEMAKER_NAME ?= ${PROJECT}-sm
SAGEMAKER_PROFILE ?= default

AWS_DEFAULT_REGION ?= us-west-2
REGION ?= ${AWS_DEFAULT_REGION}

SHMSIZE ?= 444G
WANDB_MODE ?= run
DOCKER_OPTS := \
			--name ${NAME} \
			--rm -it \
			--shm-size=${SHMSIZE} \
			-e WANDB_API_KEY \
			-e WANDB_ENTITY \
			-e WANDB_MODE \
			-e HOST_HOSTNAME= \
			-e OMP_NUM_THREADS=1 -e KMP_AFFINITY="granularity=fine,compact,1,0" \
			-e OMPI_ALLOW_RUN_AS_ROOT=1 \
			-e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
			-e NCCL_DEBUG=VERSION \
            -e DISPLAY=${DISPLAY} \
            -e XAUTHORITY \
            -e NVIDIA_DRIVER_CAPABILITIES=all \
			-v ~/.aws:/root/.aws \
			-v /root/.ssh:/root/.ssh \
			-v ~/.cache:/root/.cache \
			-v /data/:/data \
			-v /mnt/fsx/:/data2 \
			-v /dev/null:/dev/raw1394 \
			-v /mnt/fsx/tmp:/tmp \
			-v /var/run/docker.sock:/var/run/docker.sock \
			-v ${PWD}:${WORKSPACE} \
			-w ${WORKSPACE} \
			--privileged \
			--ipc=host \
			--network=host

NGPUS=$(shell nvidia-smi -L | wc -l)

all: clean

clean:
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf

docker-build:
	aws ecr get-login-password --region ${REGION} --profile ${SAGEMAKER_PROFILE} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com ; \
	docker build \
		-f docker/Dockerfile_ec2_${VERSION} \
		-t ${DOCKER_IMAGE} .

docker-build-new:
	docker build \
		-f docker/Dockerfile \
		-t ${DOCKER_IMAGE} .

docker-interactive: docker-build
	docker run --gpus all ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-interactive-new: docker-build-new
	docker run --gpus all ${DOCKER_OPTS} ${DOCKER_IMAGE} bash

docker-run: docker-build
	docker run --gpus all ${DOCKER_OPTS} ${DOCKER_IMAGE} ${COMMAND}

docker-build-sm:
	@account=$$(aws sts get-caller-identity --query Account --output text --profile ${SAGEMAKER_PROFILE}) && \
	echo $$account; \
	fullname=$$account.dkr.ecr.$(REGION).amazonaws.com/$(SAGEMAKER_NAME):latest; \
	echo $$fullname; \
	aws ecr get-login-password --region ${REGION} --profile ${SAGEMAKER_PROFILE} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com ; \
	docker build -f docker/Dockerfile_sm_${VERSION} -t $$fullname . ; \
	aws ecr create-repository --repository-name ${SAGEMAKER_NAME} --region ${REGION} --profile ${SAGEMAKER_PROFILE} > /dev/null || true ; \
	aws ecr get-login-password --region $(REGION) --profile ${SAGEMAKER_PROFILE} | docker login --username AWS --password-stdin $${fullname} ; \
	docker tag ${SAGEMAKER_NAME} $${fullname} ; \
	docker push $${fullname}

docker-interactive-sm:
	aws ecr get-login-password --region ${REGION} --profile ${SAGEMAKER_PROFILE} | docker login --username AWS --password-stdin 763104351884.dkr.ecr.${REGION}.amazonaws.com ; \
	@account=$$(aws sts get-caller-identity --query Account --output text --profile ${SAGEMAKER_PROFILE}) && \
	echo $$account; \
	fullname=$$account.dkr.ecr.$(REGION).amazonaws.com/$(SAGEMAKER_NAME):latest; \
	echo $$fullname; \
	docker run --rm -it --shm-size=${SHMSIZE} -v /data/datasets/:/opt/ml/input/data/training $$fullname /bin/bash
