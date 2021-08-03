#!/usr/bin/env bash


SOURCE_DIR="/ArcFace"

MODELNR=0000
MODELDIR="$SOURCE_DIR/model-r100-ii/model,$MODELNR"
MODELNAME="Resnet100"
DATASET="lfw"

CUDA_VISIBLE_DEVICES=6 python -u test_model.py --model $MODELDIR --model_name $MODELNAME --test_dataset $DATASET
