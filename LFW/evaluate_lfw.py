
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import warnings
from logging import warn

from collections import namedtuple

import pyeer
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
import sklearn
from scipy.optimize import brentq
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
from  matplotlib import pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from pyeer import eer_stats, plot

Stats = namedtuple('Stats', [

    # Values of EER
    'eer',
    'fmr1000',  # 1000 false match rate
    'fmr100',  # 100 false match rate
    'gmean',  # Genuine scores mean
    'imean',  # Impostor scores mean
    'FDR',
    'auc',  # Area under the ROC curve
    'fmr',
    'fnmr'
])

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds ,nrof_thresholds))
    fprs = np.zeros((nrof_folds ,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    dist = distance(embeddings1,embeddings2,distance_metric=1)
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        dist = distance(embeddings1 - mean, embeddings2 - mean, 1)
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx ,threshold_idx], fprs[fold_idx ,threshold_idx], _ = calculate_accuracy(threshold, dist[test_set], actual_issame
                                                                                                     [test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
    gmean=np.mean(1.0-dist[actual_issame==True])
    imean=np.mean(1.0-dist[actual_issame==False])
    gstd=np.std(1.0-dist[actual_issame==True])
    istd=np.std(1.0-dist[actual_issame==False])
    FDR = ((gmean-imean)**2) / (gstd**2 + istd**2)
    tpr = np.mean(tprs ,0)
    fpr = np.mean(fprs ,0)
    return tpr, fpr, accuracy,gmean,imean, FDR


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist
def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    dist = distance(embeddings1,embeddings2,1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean
def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far
def evaluate(embeddings1,embeddings2, actual_issame, nrof_folds=10, ids=""):
    # Calculate evaluation metrics
    state={}
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy ,gmean,imean, FDR= calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    auc = metrics.auc(fpr, tpr)
    ind, fmr1000 = eer_stats.get_fmr_op(fpr, 1-tpr, 0.001)
    ind, fmr100 = eer_stats.get_fmr_op(fpr, 1-tpr, 0.01)
    _,_,_,eer=eer_stats.get_eer_values(1-tpr,fpr) #1.0 - np.array(tpr)
    #thresholds = np.arange(0, 4, 0.001)
    return     Stats(eer=eer,fmr100=fmr100,fmr1000=fmr1000,gmean=gmean,imean=imean,FDR=FDR, fmr=fpr,fnmr=1-tpr,auc=auc)



def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def get_paths(lfw_dir , probe_dir, pairs, file_ext):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    embedding1=[]
    embedding2=[]
    for pair in pairs:
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(probe_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
            path1 = os.path.join(probe_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
            path_list += (path0, path1)
            emb1=np.load(path0)
            emb2=np.load(path1)
            issame_list.append(issame)
            embedding1.append(emb1)
            embedding2.append(emb2)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    print(len(issame_list))

    return path_list, issame_list ,sklearn.preprocessing.normalize(embedding1),sklearn.preprocessing.normalize(embedding2)

    
#defines first lines of a new table
def new_table(model_type, model, tablename, experiment):
    #create folder for tables
    if not os.path.exists("./tables_%s_%s/" % (model_type, experiment)):
        os.mkdir("./tables_%s_%s/" % (model_type, experiment))
        
    table = open("./tables_%s_%s/" % (model_type, experiment) + tablename + ".txt", "w")
    table.write("\\begin{table}[]\n")
    table.write("\\begin{tabular}{|l|l|l|l|l|l|l|}\n")
    table.write("\hline\n")
    table.write(model+" & EER & FMR100 & FMR1000 & G-mean & I-mean & FDR \\\ \hline\n")
    return table
    
pairs=read_pairs("pairs.txt")


def evaluate_face_mask_baseline(model="Resnet50"):
    reference_dir = "/home/aboller/ArcFace/Data/extracted_features/lfw/face_embedding/" + model
    base_dir = "./"
    mask_colors = ['black', 'lightBlue']

    states = []
    ids = []
    # Do evaluation for face-face
    path_list, issame_list, embedding1, embedding2 = get_paths(reference_dir, reference_dir, pairs, "npy")
    state = evaluate(np.array(embedding1), np.array(embedding2), issame_list, ids="")
    states.append(state)
    ids.append("LFW")
    experiment_name="Baseline-"+model
    # evaluate simulated lfw mask
    for masktype in ['a', 'b', 'c', 'd', 'e', 'f']:
        for color in mask_colors:
            probe_dir = "/home/aboller/ArcFace/Data/extracted_features/lfw/mask_" + color + "_" + masktype + "/" + model
            path_list, issame_list, embedding1, embedding2 = get_paths(reference_dir, probe_dir, pairs, "npy")
            state = evaluate(np.array(embedding1), np.array(embedding2), issame_list, ids="")
            states.append(state)
            ids.append("SM-" + masktype.upper() + "-" + color)
            # create subplot folder
    if not os.path.exists(os.path.join(base_dir, model+ "_baseline")):
         os.mkdir(os.path.join(base_dir, model+ "_baseline"))
    plot.plt_roc_curve(states, ids, save_path=os.path.join(base_dir, model+"_baseline"),ext=experiment_name + ".eps")
    
    #create latex table
    if model == "Resnet50":
        model_type = "r50"
    else:
        model_type = "m"
    table = new_table(model_type, model, experiment_name, "Baseline")

    # filter best results
    best_eer = 1
    best_fmr100 = 1
    best_fmr1000 = 1
    best_FDR = 0
    for state in states:
        if best_eer > state.eer:
            best_eer = state.eer
        if best_fmr100 > state.fmr100:
            best_fmr100 = state.fmr100
        if best_fmr1000 > state.fmr1000:
            best_fmr1000 = state.fmr1000
        if best_FDR < state.FDR:
            best_FDR = state.FDR

    for index, st in enumerate(states):
        # compute percentage values
        # compute percentage
        eer = st.eer
        eer *= 100
        eer = round(eer, 4)
        fmr100 = st.fmr100
        fmr100 *= 100
        fmr100 = round(fmr100, 4)
        fmr1000 = st.fmr1000
        fmr1000 *= 100
        fmr1000 = round(fmr1000, 4)
        FDR = round(st.FDR, 4)
        # set best values
        if st.eer == best_eer:
            eer = "\\textbf{" + str(eer) + "}"
        else:
            eer = str(eer)
        if st.fmr100 == best_fmr100:
            fmr100 = "\\textbf{" + str(fmr100) + "}"
        else:
            fmr100 = str(fmr100)
        if st.fmr1000 == best_fmr1000:
            fmr1000 = "\\textbf{" + str(fmr1000) + "}"
        else:
            fmr1000 = str(fmr1000)
        if st.FDR == best_FDR:
            FDR = "\\textbf{" + str(FDR) + "}"
        else:
            FDR = str(FDR)

        # experiment, EER, FMR100, FMR1000, G-mean, I-mean, FDR
        table.write(ids[index] + " & " + eer + "\% & " + fmr100 + "\% & " + fmr1000 + "\% & " + str(
            round(st.gmean, 4)) + " & " + str(round(st.imean, 4)) + " & " + FDR + " \\\ \hline\n")
    # add last lines of latex table
    table.write("\end{tabular}\n")
    table.write("\caption{" + experiment_name + "}\n")
    table.write("\end{table}")
    table.close()

def evaluate_mask_FB(model="Resnet50", maskreference=False):
 reference_dir="/home/aboller/ArcFace/Data/extracted_features/lfw/face_embedding/"+model
 base_dir="./"
 mask_colors = ['black', 'lightBlue']
 frames=[0,1,2,4,8,16,32]

    
 for masktype in ['a', 'b', 'c', 'd', 'e', 'f']:
    states=[]
    for color in mask_colors:
        states = []
        ids=[]
        probe_dir = "/home/aboller/ArcFace/Data/extracted_features/lfw/mask_"+color+"_"+masktype+"/"+model
        if(maskreference):
            reference_dir=probe_dir
        path_list, issame_list, embedding1, embedding2 = get_paths(reference_dir, probe_dir, pairs, "npy")
        state = evaluate(np.array(embedding1), np.array(embedding2), issame_list, ids="")
        states.append(state)
        ids.append("BS")
        for frame in frames:
            probe_dir = "/home/fboutros/face_mask/WeightedMaskTripletLoss/outputlwf-Resnet50/"+str(frame)+"/mask_"+color+"_"+masktype+"/"+model+str(frame)
            if (maskreference):
                reference_dir = probe_dir
            path_list, issame_list, embedding1, embedding2 = get_paths(reference_dir, probe_dir, pairs, "npy")
            state = evaluate(np.array(embedding1), np.array(embedding2), issame_list, ids="")
            states.append(state)
            if frame == 0:
                curve_name = "COS"
            else:
                curve_name = "HCOS"+str(frame)
            ids.append(curve_name)
        
        if (maskreference):
            model_name = model+"SM-SM"
            experiment_name = "SM-SM"+masktype.upper()+"-"+color
        else:
            model_name = model+"F-SM"
            experiment_name = "F-SM"+masktype.upper()+"-"+color
        #create subplot folder
        save_dir=os.path.join(base_dir, model_name, masktype+"_"+color)
        if not os.path.exists(os.path.join(base_dir, model_name)):
            os.mkdir(os.path.join(base_dir, model_name))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        plot.plt_roc_curve(states, ids, save_path=save_dir, ext=experiment_name+".eps")
        #create a latex table
        if model == "Resnet50":
            model_type = "r50"
        else:
            model_type = "m"
        if (maskreference):
            exp = "mask-mask"
        else: exp = "face-mask"
        table = new_table(model_type, model, experiment_name, exp)
        
        #filter best results
        best_eer = 1
        best_fmr100 = 1
        best_fmr1000 = 1
        best_FDR = 0
        for state in states:
            if best_eer > state.eer:
                best_eer = state.eer
            if best_fmr100 > state.fmr100:
                best_fmr100 = state.fmr100
            if best_fmr1000 > state.fmr1000:
                best_fmr1000 = state.fmr1000
            if best_FDR < state.FDR:
                best_FDR = state.FDR
        
        for index, st in enumerate(states):
            #compute percentage values
            #compute percentage
            eer = st.eer
            eer *= 100
            eer = round(eer,4)
            fmr100 = st.fmr100
            fmr100 *= 100
            fmr100 = round(fmr100,4)
            fmr1000 = st.fmr1000
            fmr1000 *= 100
            fmr1000 = round(fmr1000,4)
            FDR = round(st.FDR,4)
            #set best values
            if st.eer == best_eer:
                eer = "\\textbf{"+str(eer)+"}"
            else: eer = str(eer)
            if st.fmr100 == best_fmr100:
                fmr100 = "\\textbf{"+str(fmr100)+"}"
            else: fmr100 = str(fmr100)
            if st.fmr1000 == best_fmr1000:
                fmr1000 = "\\textbf{"+str(fmr1000)+"}"
            else: fmr1000 = str(fmr1000)
            if st.FDR == best_FDR:
                FDR = "\\textbf{"+str(FDR)+"}"
            else: FDR = str(FDR)
                
            # experiment, EER, FMR100, FMR1000, G-mean, I-mean, FDR
            table.write(ids[index] + " & " + eer + "\% & " + fmr100 + "\% & " + fmr1000 + "\% & " + str(round(st.gmean,4)) + " & " + str(round(st.imean,4)) + " & " + FDR + " \\\ \hline\n")
        #add last lines of latex table
        table.write("\end{tabular}\n")
        table.write("\caption{"+experiment_name+"}\n")
        table.write("\end{table}")
        table.close()
def parse_args():
  parser = argparse.ArgumentParser(description='evaluate simulated mask on LFW')
  parser.add_argument('--model', default="Resnet50", help='model name')
  parser.add_argument('--evalset', default=0, help='evaluation type')
  return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    if(int(args.evalset)==1):
        evaluate_face_mask_baseline(args.model)
    elif(int(args.evalset)==2):
        print("Face-Mask evaluation")
        evaluate_mask_FB(args.model,False) # Do evaluation for Face-mask
    elif(int(args.evalset)==3):
        print("Mask-Mask evaluation")
        evaluate_mask_FB(args.model,True) # Do evaluation for mask-mask




