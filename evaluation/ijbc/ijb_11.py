# coding: utf-8

import os
import numpy as np
#import cPickle
import pickle
import pandas as pd
import matplotlib
import torch

from model.model import SingleLayerModel

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
from sklearn import preprocessing
import sys
sys.path.append('./')

import argparse
import cv2
import numpy as np
import mxnet as mx
from skimage import transform as trans
import sklearn

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='do ijb test')
# general
parser.add_argument('--model-prefix', default='pretrained-model/model-r50/model,0129', help='path to load model.')
parser.add_argument('--model-epoch', default=2, type=int, help='')
parser.add_argument('--gpu', default=6, type=int, help='gpu id')
parser.add_argument('--batch-size', default=512, type=int, help='')
parser.add_argument('--embedding-size', default=512, type=int, help='')
parser.add_argument('--job', default='r50-m-m', type=str, help='job name')
parser.add_argument('--save_path', default='./result_r50_mr_mp', type=str, help='job name')

parser.add_argument('--image-path', default='./ijbc/loose_crop', type=str, help='job name')
parser.add_argument('--image-masked-path', default='./ijbc/loose_crop_mask', type=str, help='job name')

parser.add_argument('--weights', default='weights/weights-SRT-ResNet50/weights.pt', help='')
parser.add_argument('--masked-reference', default=True, help='')
parser.add_argument('--masked-probe', default=True, help='')
parser.add_argument('--eum',default=True, help='process embedding with EUM model')

parser.add_argument('--target',
                    default='ijbc',
                    type=str,
                    help='target, set to ijbc or IJBB')

args = parser.parse_args()

target = args.target
img_path = args.image_path
img_masked_path=args.image_masked_path

model_path = args.model_prefix
gpu_id = args.gpu
epoch = args.model_epoch
use_norm_score = True  # if Ture, TestMode(N1)
use_detector_score = True  # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
save_path =args.save_path

job = args.job
class Embedding:
  def __init__(self, prefix, epoch, ctx_id=0):
    print('loading',prefix, epoch)
    ctx = mx.gpu(ctx_id)
    prefix, epoch = prefix.split(",")
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,int(epoch))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    image_size = (112,112)
    self.image_size = image_size
    model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
    model.bind(for_training=False, data_shapes=[('data', (2, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    self.model = model
    self.cnn = SingleLayerModel(embedding_size=args.embedding_size).cuda()
    self.cnn.load_state_dict(torch.load(args.weights))
    self.cnn.eval()
    src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041] ], dtype=np.float32 )
    src[:,0] += 8.0
    self.src = src

  def get(self, rimg, landmark,isMask=False):
    assert landmark.shape[0]==68 or landmark.shape[0]==5
    assert landmark.shape[1]==2
    if landmark.shape[0]==68:
      landmark5 = np.zeros( (5,2), dtype=np.float32 )
      landmark5[0] = (landmark[36]+landmark[39])/2
      landmark5[1] = (landmark[42]+landmark[45])/2
      landmark5[2] = landmark[30]
      landmark5[3] = landmark[48]
      landmark5[4] = landmark[54]
    else:
      landmark5 = landmark
    tform = trans.SimilarityTransform()
    tform.estimate(landmark5, self.src)
    M = tform.params[0:2,:]
    img = cv2.warpAffine(rimg,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flip = np.fliplr(img)
    img = np.transpose(img, (2,0,1)) #3*112*112, RGB
    img_flip = np.transpose(img_flip,(2,0,1))
    input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]),dtype=np.uint8)
    input_blob[0] = img
    input_blob[1] = img_flip
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    feat = self.model.get_outputs()[0].asnumpy()

    if (isMask and args.eum):
        with torch.no_grad():
            feat = sklearn.preprocessing.normalize(feat)
            feat = self.cnn(torch.tensor(feat).cuda().float()).detach().cpu().numpy()
    feat = feat.reshape([-1, feat.shape[0] * feat.shape[1]])
    feat = feat.flatten()

    return feat

def read_template_media_list(path):
    #ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    #pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    #print(pairs.shape)
    #print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# In[ ]:


def get_image_feature(img_path, img_list_path, model_path, epoch, gpu_id, isMask=False):
    img_list = open(img_list_path)
    embedding = Embedding(model_path, epoch, gpu_id)
    files = img_list.readlines()
    print('files:', len(files))
    faceness_scores = []
    img_feats = []
    for img_index, each_line in enumerate(files):
        if img_index % 500 == 0:
            print('processing', img_index)
        name_lmk_score = each_line.strip().split(' ')
        img_name = os.path.join(img_path, name_lmk_score[0])
        img = cv2.imread(img_name)
        lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                       dtype=np.float32)
        lmk = lmk.reshape((5, 2))
        img_feats.append(embedding.get(img, lmk,isMask))
        faceness_scores.append(name_lmk_score[-1])
    img_feats = np.array(img_feats).astype(np.float32)
    faceness_scores = np.array(faceness_scores).astype(np.float32)

    #img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
    #faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
    return img_feats, faceness_scores


# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):
        (ind_t, ) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m, ) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    #template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    #print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

def verification(template_norm_feats=None,template_norm_feats_mask=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1), ))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats_mask[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score

# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1), ))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


# # Step1: Load Meta Data

# In[ ]:

assert target == 'ijbc' or target == 'IJBB'

# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
start = timeit.default_timer()
templates, medias = read_template_media_list(
    os.path.join('%s/meta' % target, '%s_face_tid_mid.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:

# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
p1, p2, label = read_template_pair_list(
    os.path.join('%s/meta' % target,
                 '%s_template_pair_label.txt' % target.lower()))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 2: Get Image Features

# In[ ]:

# =============================================================
# load image features
# format:
#           img_feats: [image_num x feats_dim] (227630, 512)
# =============================================================
start = timeit.default_timer()
img_list_path = './%s/meta/%s_name_5pts_score.txt' % (target, target.lower())
if (args.masked_referene and args.masekd_probe):
    img_feats, faceness_scores = get_image_feature(img_masked_path, img_list_path,
                                               model_path, epoch, gpu_id,args.masked_reference)
elif (not args.masked_referene and args.masekd_probe):
    img_feats, faceness_scores = get_image_feature(img_path, img_list_path,
                                                   model_path, epoch, gpu_id, args.masked_reference)
    img_feats_mask, faceness_scores = get_image_feature(img_masked_path, img_list_path,
                                                        model_path, epoch, gpu_id, isMask=True)

elif (not args.masked_referene and not args.masekd_probe):
    img_feats, faceness_scores = get_image_feature(img_path, img_list_path,
                                                   model_path, epoch, gpu_id, args.masked_reference)

stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                          img_feats.shape[1]))


# # Step3: Get Template Features

# In[ ]:

# =============================================================
# compute template features from image features.
# =============================================================
start = timeit.default_timer()
# ==========================================================
# Norm feature before aggregation into template feature?
# Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
# ==========================================================
# 1. FaceScore （Feature Norm）
# 2. FaceScore （Detector）

if use_flip_test:
    # concat --- F1
    #img_input_feats = img_feats
    # add --- F2
    img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                2] + img_feats[:, img_feats.shape[1] // 2:]
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_feats_mask[:, 0:img_feats_mask.shape[1] //
                                               2] + img_feats_mask[:, img_feats_mask.shape[1] // 2:]
else:
    img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_feats_mask[:, 0:img_feats_mask.shape[1] // 2]


if use_norm_score:
    img_input_feats = img_input_feats
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_input_feats_mask

else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats**2, -1, keepdims=True))
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_input_feats_mask / np.sqrt(np.sum(img_input_feats_mask ** 2, -1, keepdims=True))

if use_detector_score:
    print(img_input_feats.shape, faceness_scores.shape)
    #img_input_feats = img_input_feats * np.matlib.repmat(faceness_scores[:,np.newaxis], 1, img_input_feats.shape[1])
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_input_feats_mask * faceness_scores[:, np.newaxis]

else:
    img_input_feats = img_input_feats
    if  (not args.masked_referene and args.masekd_probe):
        img_input_feats_mask = img_input_feats_mask

template_norm_feats, unique_templates = image2template_feature(
    img_input_feats, templates, medias)
if (not args.masked_referene and args.masekd_probe):
    template_norm_feats_mask, unique_templates_mask = image2template_feature(
    img_input_feats_mask, templates, medias)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# # Step 4: Get Template Similarity Scores

# In[ ]:

# =============================================================
# compute verification mfr_evaluation between template pairs.
# =============================================================
start = timeit.default_timer()
if (not args.masked_referene and args.masekd_probe):
    score = verification(template_norm_feats, template_norm_feats_mask, unique_templates, p1, p2)
else:
    score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

# In[ ]:


if not os.path.exists(save_path):
    os.makedirs(save_path)

score_save_file = os.path.join(save_path, "%s.npy" % job)
np.save(score_save_file, score)

# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))

fig = plt.figure()
for method in methods:
    sr = scores[method]
    genuine_score = sr[label == 1]
    imposter_score = sr[label == 0]
    with open(save_path+ "/genuine.txt", "w") as f:
        for g in genuine_score:
            f.write(str(g) + "\n")
    with open(save_path + "/imposter.txt", "w") as f:
        for i in imposter_score:
            f.write(str(i) + "\n")
