#!/usr/bin/env bash


SOURCE_DIR="ArcFace"

MODELNR=0096
MODELDIR="$SOURCE_DIR/models/model-mobilefacenet/model,$MODELNR"
MODELNAME="Mobilefacenet"
DATASET="lfw"

CUDA_VISIBLE_DEVICES=0 python -u test_model.py --model $MODELDIR --model_name $MODELNAME --test_dataset $DATASET
