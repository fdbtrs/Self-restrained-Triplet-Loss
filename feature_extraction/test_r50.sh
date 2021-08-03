#!/usr/bin/env bash


SOURCE_DIR="ArcFace"

MODELNR=0129
MODELDIR="$SOURCE_DIR/models/model-r50/model,$MODELNR"
MODELNAME="Resnet50"
DATASET="lfw"

CUDA_VISIBLE_DEVICES=6 python -u test_model.py --model $MODELDIR --model_name $MODELNAME --test_dataset $DATASET
