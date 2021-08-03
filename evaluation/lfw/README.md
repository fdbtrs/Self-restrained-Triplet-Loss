# MFR

# Evaluation of synthetically generated masked face of LFW

### Synthetically generated masked- The mask color and type are randomly applied to each image of LFW
- Download LFW dataset on your own and strictly follow the licence distribution 
- Run align_lfw.py --> This script  aligns and crops lfw images and creates new version with simulated mask 
Note: to regenerate the result in the paper, please use the exact mask type color and type provided in LFW-log.txt

### Evaluation
- Extract the face embedding using ResNet-100, ResNet-50 or MobileFaceNet 
- The script for each of there models are under feature_extraction directory

- Run main.py to process the embedding with SRT solution

### Plot the result
- Run plot.py



