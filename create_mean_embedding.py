import os
import sys
import threading
import time
import numpy as np
from  sklearn.metrics.pairwise import cosine_similarity

embedding_size=128
target='/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/M1P-avg1-32'
os.mkdir(target)
embedding_dirs=['/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/0/M1P0','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/1/M1P1','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/2/M1P2','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/4/M1P4','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/8/M1P8','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/16/M1P16','/home/fboutros/face_mask/WeightedMaskTripletLoss/output1m/32/M1P32']
list_embedding_file=os.listdir(embedding_dirs[0])
for embedding_file in list_embedding_file:
    avg_embedding=np.zeros((len(embedding_dirs),embedding_size))
    for i  in range(0,len(embedding_dirs)):
        avg_embedding[i,:]= np.load(os.path.join(embedding_dirs[i],embedding_file))

    np.save(os.path.join(target,embedding_file),np.mean(avg_embedding,axis=0))