#!/bin/bash
ECR=965350412536.dkr.ecr.us-west-2.amazonaws.com
IMAGE_TAG=betteromics-python-slim:cedars_genomics
echo "Building Image"
sudo docker build -t $IMAGE_TAG .
echo "Tagging Image"
sudo docker tag $IMAGE_TAG $ECR/$IMAGE_TAG
echo "Pushing Image"
sudo docker push $ECR/$IMAGE_TAG
