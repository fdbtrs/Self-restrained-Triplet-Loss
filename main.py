import pdb

from torch.backends import cudnn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from tqdm import tqdm
import torch

import numpy as np
import os
import time
import torch.nn.functional as F
import argparse
from pathlib import Path

import torch.nn as nn
import adabound

from util.databaseTest import MaskDatasetTestMFR2
from model.model import SingleLayerModel
from util.losses import TripletLoss
from util.database_triplet import MaskDataset
from util.databaseTest import MaskDatasetTest
from util.misc import CSVLogger

def setupt():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
    #torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    # Data Loader (Input Pipeline)
def CosineDistance(x1,x2):
    return 1- F.cosine_similarity(x1,x2)


metric= nn.CosineSimilarity(eps=1e-6)
'''
def metric(emb1,emb2):
     sub=torch.sub(emb1,emb2)
     sm=torch.sum(sub*sub,dim=1)
     return torch.norm(emb1 - emb2, 2, 1).detach().cpu().numpy()
'''
cnn = SingleLayerModel(embedding_size=512).cuda()



def validation(val_loader):
    cnn.eval()
    scores=[]
    scores_imposter=[]
    i=200
    for mask_embedding,face_embedding,negative_embedding,cls,_ in val_loader:
        mask_embedding = mask_embedding.cuda()
        face_embedding = face_embedding.cuda()
        negative_embedding = negative_embedding.cuda()
        with torch.no_grad():
            pred= cnn(mask_embedding)
        scores.append(metric(l2_norm(pred),l2_norm(face_embedding)).item())
        m = (metric(l2_norm(pred) , l2_norm(negative_embedding)).item())
        scores_imposter.append(m )
        i=i-1
    cnn.train()


    return np.mean(scores),np.mean(scores_imposter)

def l2_norm( input, axis=1):
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output
def validation_init(val_loader):
    cnn.eval()
    scores=[]
    scores_imposter=[]
    i=200
    for mask_embedding,face_embedding,negative_embedding,cls,_ in val_loader:
        mask_embedding = mask_embedding.cuda()
        face_embedding = face_embedding.cuda()
        negative_embedding = negative_embedding.cuda()
        scores.append(metric(l2_norm(mask_embedding),l2_norm(face_embedding)).item())
        m=metric(l2_norm(mask_embedding),l2_norm(negative_embedding)).item()
        scores_imposter.append(m)
        i=i-1

    return np.mean(scores), np.mean(scores_imposter)



def training(args):
    if not os.path.isdir('logs'):
        os.makedirs('logs')
    train_loader = torch.utils.data.DataLoader(dataset=MaskDataset(root=args.data_dir,random=True,isTraining=True),
                                               batch_size=int(512),
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=16)

    val_loader = torch.utils.data.DataLoader(
        dataset=MaskDataset(root=args.data_dir+'validation/',random=True,isTraining=False),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=2)
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=float(0.1), momentum=0.9, nesterov=True,
                                    weight_decay=0.0)  # 0.0001
    scheduler = StepLR(cnn_optimizer, gamma=0.1, step_size=3)
    criterion=TripletLoss(distance=args.loss).cuda()
    early_stopping = True
    patience = 20

    epochs_no_improvement = 0
    max_val_fscore = 0.0
    best_weights = None
    best_epoch = -1
    filename = 'logs/' + str(args.loss) + '.csv'
    csv_logger = CSVLogger(args=None, fieldnames=['epoch', 'TotalLoss', 'positive_loss','negative_loss','negative_positive', 'val_acc'], filename=filename)
    init_val_fscore, val_fscore_imposter = validation_init(val_loader)
    # set model to train mode
    cnn.train()

    tqdm.write('genuine: %.5f' % (init_val_fscore))
    tqdm.write('imposter: %.5f' % (val_fscore_imposter))
    update_weight_loss=True
    val_fscore=0.
    for epoch in range(1, 1 + args.epoch):
        loss_total = 0.
        fscore_total = 0.
        positive_loss_totoal=0.
        negative_loss_total=0.
        negative_positive_total=0.

        progress_bar = tqdm(train_loader)
        for i, (mask_embedding,face_embedding,negative_embedding,label,_) in enumerate(progress_bar):
            progress_bar.set_description('Epoch ' + str(epoch))

            mask_embedding = mask_embedding.cuda()
            face_embedding =face_embedding.cuda()
            negative_embedding=negative_embedding.cuda()
            label=label.cuda()
            cnn.zero_grad()
            pred = cnn(mask_embedding)
            loss, positive_loss,negative_loss , negative_positive= criterion(pred, face_embedding, negative_embedding)
            loss.backward()
            cnn_optimizer.step()

            loss_total += loss.item()
            positive_loss_totoal+=positive_loss.item()
            negative_loss_total+=negative_loss.item()
            negative_positive_total+=negative_positive.item()

            row = {'epoch': str(epoch)+str("-")+str(i), 'TotalLoss': str(loss_total / (i + 1)), 'positive_loss': str(positive_loss_totoal / (i + 1)), 'negative_loss': str(negative_loss_total / (i + 1)),'negative_positive':str(negative_positive_total / (i + 1)),'val_acc':str(val_fscore)}
            csv_logger.writerow(row)


            progress_bar.set_postfix(
                 loss='%.5f' % (loss_total / (i + 1)),negative_loss='%.5f' % (negative_loss_total/(i+1) ),positive_loss='%.5f' % (positive_loss_totoal / (i + 1)),negative_positive='%.5f' % (negative_positive_total / (i + 1)) )


        val_fscore ,val_fscore_imposter= validation(val_loader)

        tqdm.write('fscore: %.5f' % (val_fscore))
        tqdm.write('imposter: %.5f' % (val_fscore_imposter))

        # scheduler.step(epoch)  # Use this line for PyTorch <1.4
        scheduler.step()  # Use this line for PyTorch >=1.4

        #row = {'epoch': str(epoch), 'train_acc': str(train_fscore), 'val_acc': str(val_fscore)}
        #csv_logger.writerow(row)
        do_stop=False
        if early_stopping:
            if val_fscore > max_val_fscore:
                max_val_fscore = val_fscore
                epochs_no_improvement = 0
                best_weights = cnn.state_dict()
                best_epoch = epoch
            else:
                epochs_no_improvement += 1

            if epochs_no_improvement >= patience and do_stop:
                print(f"EARLY STOPPING at {best_epoch}: {max_val_fscore}")
                break
        else:
            best_weights = cnn.state_dict()
    if not os.path.isdir(os.path.join(args.weights,str(args.loss))):
        os.makedirs(os.path.join(args.weights,str(args.loss)))
    torch.save(best_weights, os.path.join(args.weights,str(args.loss),'weights.pt'))
    csv_logger.close()



def testing(args):
    cnn.load_state_dict(torch.load(os.path.join(args.weights,str(args.loss),'weights.pt')))
    cnn.eval()
    if not os.path.isdir(args.test_output):
        os.makedirs(args.test_output)
    if(args.do_test_ar):
        test_data = args.test_dir_ar.split(',')
    else:
        test_data=args.test_dir.split(',')
    for t in test_data:
        save_path=os.path.join(args.test_output,str(args.loss),os.path.basename(t)+str(args.loss))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if (args.do_test_ar):
            test_loader = torch.utils.data.DataLoader(dataset=MaskDatasetTestMFR2(root=t),batch_size=1, shuffle=False, pin_memory=True,num_workers=2)
        else:
            test_loader = torch.utils.data.DataLoader(
            dataset=MaskDatasetTest(root=t),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=2)
        for mask_embedding,f,_ in test_loader:
         mask_embedding = mask_embedding.cuda()
         with torch.no_grad():
            pred = cnn(mask_embedding)
         pred = pred.squeeze(dim=0).detach().cpu().numpy()
         f = f[0]
         #print(f)
         if (args.do_test_ar):
             if not os.path.isdir(save_path+'/'+f.split('_')[0]):
                 os.makedirs(save_path+'/'+f.split('_')[0])
             np.save(os.path.join(save_path+'/'+f.split('_')[0],f),pred)
         else:
            np.save(os.path.join(save_path, f), pred)



def testlfw(args):
    cnn.load_state_dict(torch.load(os.path.join(args.weights,str(args.loss),'weights.pt')))
    cnn.eval()
    if not os.path.isdir(args.test_output):
        os.makedirs(args.test_output)

    test_data=args.test_dir_lfw
    for t in test_data:
        path = t.split("/")
        save_path=os.path.join(args.lfw_test_output,str(args.loss),path[len(path)-2],os.path.basename(t)+str(args.loss))
        print(t)
        print(save_path)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        test_loader = torch.utils.data.DataLoader(dataset=MaskDatasetTestMFR2(root=t),batch_size=1, shuffle=False, pin_memory=True,num_workers=2)
        for mask_embedding,f,dr in test_loader:
         mask_embedding = mask_embedding.cuda()
         with torch.no_grad():
            pred = cnn(mask_embedding)
         pred = pred.squeeze(dim=0).detach().cpu().numpy()
         f = f[0]
         if not os.path.isdir(save_path+'/'+dr[0]):
                 os.makedirs(save_path+'/'+dr[0])
         np.save(os.path.join(save_path+'/'+dr[0],f),pred)

def load_weight(weights):
    cnn.load_state_dict(torch.load(weights))


def parse_args():
  parser = argparse.ArgumentParser(description='Train face mask adaption')
  parser.add_argument('--loss', default="SRT", help='loss Triplet or SRT')
  parser.add_argument('--mode', default=2, help='')
  parser.add_argument('--weights', default='weights/weightsResNet100', help='')
  parser.add_argument('--epoch', default=10, help='')
  parser.add_argument('--data_dir', default="ms1m_features_dlib_r100/", help='training dataset directory')

  parser.add_argument('--test_dir', default='maskfilm_dataset/ResNet100/M12P,maskfilm_dataset/ResNet100/M12R', help='')


  parser.add_argument('--test_dir_ar',default="extracted_features/mfr2/Resnet100")
  parser.add_argument('--do_test_ar',default=False)
  parser.add_argument('--test_lfw',default=False)
  parser.add_argument('--test_dir_lfw',default="extracted_features/lfw/face_embedding/Resnet100")

  parser.add_argument('--lfw_test_output', default='outputlwf-Resnet100/', help='')

  parser.add_argument('--test_output', default='outputResNet100/', help='')

  args = parser.parse_args()
  return args

def parse_args_ResNet50():
  parser = argparse.ArgumentParser(description='Train face mask adaption')
  parser.add_argument('--loss', default="SRT", help='loss Triplet or SRT')
  parser.add_argument('--mode', default=2, help='')
  parser.add_argument('--weights', default='weights/weightsResNet50', help='')
  parser.add_argument('--epoch', default=10, help='')
  parser.add_argument('--data_dir', default="ms1m_features_dlib_r50/", help='training dataset directory')

  parser.add_argument('--test_dir', default='maskfilm_dataset/ResNet50/M12P,maskfilm_dataset/ResNet50/M12R', help='')


  parser.add_argument('--test_dir_ar',default="extracted_features/mfr2/Resnet50")
  parser.add_argument('--do_test_ar',default=False)
  parser.add_argument('--test_lfw',default=False)
  parser.add_argument('--test_dir_lfw',default="extracted_features/lfw/face_embedding/Resnet50")

  parser.add_argument('--lfw_test_output', default='outputlwf-Resnet50/', help='')

  parser.add_argument('--test_output', default='outputResNet50/', help='')

  args = parser.parse_args()
  return args

def parse_args_MobilefaceNet():
  parser = argparse.ArgumentParser(description='Train face mask adaption')
  parser.add_argument('--loss', default="SRT", help='loss Triplet or SRT')
  parser.add_argument('--mode', default=2, help='')
  parser.add_argument('--weights', default='weights/weightsResNet50', help='')
  parser.add_argument('--epoch', default=10, help='')
  parser.add_argument('--data_dir', default="ms1m_features_dlib_MobilefaceNet/", help='training dataset directory')

  parser.add_argument('--test_dir', default='maskfilm_dataset/MobilefaceNet/M12P,maskfilm_dataset/MobilefaceNet/M12R', help='')


  parser.add_argument('--test_dir_ar',default="extracted_features/mfr2/MobilefaceNet")
  parser.add_argument('--do_test_ar',default=False)
  parser.add_argument('--test_lfw',default=False)
  parser.add_argument('--test_dir_lfw',default="extracted_features/lfw/face_embedding/MobilefaceNet")

  parser.add_argument('--lfw_test_output', default='outputlwf-MobilefaceNet/', help='')

  parser.add_argument('--test_output', default='outputMobilefaceNet/', help='')

  args = parser.parse_args()
  return args




if __name__ == '__main__':
    args=parse_args()
    if(args.mode==0):
        training(args)
    elif(args.mode==1):
        testing(args)
    elif  (args.mode==2):
        testlfw(args)
    elif (args.mode==3):
        args.do_test_ar = True
        testing(args)

