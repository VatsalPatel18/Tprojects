#!/bin/bash

set -exou pipefail

# will generate a tag
source $BETTER_HOME/root/dataset_manager/deployment/get_tag.sh

IMAGE_TAG=$ECR_REPO/tximport:$TAG

aws ecr get-login-password | docker login --password-stdin  --username AWS $ECR_REPO

echo "Building Image"
docker build -t $IMAGE_TAG .

echo "Pushing Image"
docker push $IMAGE_TAG
