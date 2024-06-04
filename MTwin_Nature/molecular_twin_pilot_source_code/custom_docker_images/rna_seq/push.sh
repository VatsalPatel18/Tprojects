#!/bin/bash

set -exou pipefail

ECR=965350412536.dkr.ecr.us-west-2.amazonaws.com
IMAGE_TAG=$ECR/kallisto:cedars_rna_seq_v2

aws ecr get-login-password | docker login --password-stdin  --username AWS $ECR_REPO

echo "Building Image"
docker build -t $IMAGE_TAG .
echo "Pushing Image"
docker push $IMAGE_TAG
