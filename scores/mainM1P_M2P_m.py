import argparse
import os
import sys
import threading
import time
import numpy as np
import sklearn
from  sklearn.metrics.pairwise import cosine_similarity


def norm(x):
    return x.reshape(1, -1)
    #return sklearn.preprocessing.normalize(x.reshape(1, -1))

def calc_score(ref,probe,target_path):
    set_name = str(os.path.basename(os.path.normpath(ref))) + "-" + str(os.path.basename(os.path.normpath(probe)))
    if not os.path.exists(os.path.join(target_path, set_name)):
        os.makedirs(os.path.join(target_path, set_name))
    scores_imposter = open(os.path.join(target_path, set_name, 'imposter.txt'), 'a')
    scores_genuine = open(os.path.join(target_path, set_name, 'genuine.txt'), 'a')
    for refr, refd, reffiles in os.walk(ref):
        for rfile in reffiles:
            ref_id = (rfile.split('_')[0]).split('D')[1]
            ref_feature = np.load(os.path.join(refr, rfile))
            ref_feature=norm(ref_feature)
            for probr, probd, probfiles in os.walk(probe):
                for pfile in probfiles:
                    probe_id = (pfile.split('_')[0]).split('D')[1]
                    probe_feature = np.load(os.path.join(probr, pfile))
                    probe_feature=norm(probe_feature)
                    sim = cosine_similarity(ref_feature, probe_feature)
                    if int(ref_id) != int(probe_id):
                        scores_imposter.write(str(sim[0][0]) + '\n')
                    elif int(ref_id) == int(probe_id):
                        scores_genuine.write(str(sim[0][0]) + '\n')

    scores_imposter.close()
    scores_genuine.close()


def getcomparsionScores(args):
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    ref_list= args.ref_list.split(',')
    probe_list=args.probe_list.split(',')
    for ref in ref_list:
        for probe in probe_list:
            calc_score(ref, probe,args.target_path)
    '''
    ref_list = args.ref_list1.split(',')
    probe_list = args.probe_list1.split(',')
    for ref in ref_list:
        for probe in probe_list:
            calc_score(ref, probe,args.target_path)
    ref_list = args.ref_list2.split(',')
    probe_list = args.probe_list2.split(',')
    for ref in ref_list:
        for probe in probe_list:
            calc_score(ref, probe,args.target_path)
    ref_list = args.ref_list3.split(',')
    probe_list = args.probe_list3.split(',')
    for ref in ref_list:
        for probe in probe_list:
            calc_score(ref, probe,args.target_path)
    '''


def parse_args():
  parser = argparse.ArgumentParser(description='comparsion sores  ')
  parser.add_argument('--main_dir', default="/home/fboutros/PR/extracted_features/maskfilm_dataset", help='feature dir')
  parser.add_argument('--target_path', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/similarities_cosine/ResNet100", help='feature dir')
  parser.add_argument('--ref_list', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/BLR", help='feature dir')
  #parser.add_argument('--probe_list', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/M12P,/home/fboutros/PR/SRT/output2cr/SRT/M12PSRT,/home/fboutros/PR/SRT/output2cr/Triplet/M12PTriplet,/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/BLP", help='feature dir')
  parser.add_argument('--probe_list', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/M12P", help='feature dir')

  parser.add_argument('--ref_list1', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/M12R", help='feature dir')
  parser.add_argument('--probe_list1', default="/home/fboutros/PR/extracted_features/maskfilm_dataset/ResNet100/M12P", help='feature dir')
  parser.add_argument('--ref_list2', default="/home/fboutros/PR/SRT/output2cr/SRT/M12RSRT", help='feature dir')
  parser.add_argument('--probe_list2', default="/home/fboutros/PR/SRT/output2cr/SRT/M12PSRT", help='feature dir')
  parser.add_argument('--ref_list3', default="/home/fboutros/PR/SRT/output2cr/Triplet/M12RSRT", help='feature dir')
  parser.add_argument('--probe_list3', default="/home/fboutros/PR/SRT/output2cr/Triplet/M12PSRT", help='feature dir')
  args = parser.parse_args()
  return args
if __name__ == '__main__':
    args=parse_args()
    getcomparsionScores(args)


