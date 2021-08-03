import os
import sys
source_path = os.path.dirname(__file__)
import feature_extraction.face_model
import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--model_name', default='', help='Name of the target model folder.')
parser.add_argument('--test_dataset', default='', help='Name of the dataset for testing.')
args = parser.parse_args()

model_name = args.model_name
paths = []
#set correct database and output paths
if args.test_dataset == "maskfilm":
    print("Testing on maskfilm dataset")
    paths = ["Data/maskedface"]
    output_dir = ["extracted_features/maskfilm_dataset"]
elif args.test_dataset=="mfr2":
  paths = ["data/mfr2/mfr2_aligned/mfr2"]
  output_dir = ["extracted_features/mfr2"]
elif args.test_dataset=="lfw":
    paths=["data/lfw_aligned"]
    output_dir = ["extracted_features/lfw/face_embedding"]
    paths.append("data/lfw_aligned_mask_random")
    output_dir.append("extracted_features/lfw/face_embedding_mask")



else: sys.exit("Wrong database.")


model = face_model.FaceModel(args)
for index, path in enumerate(paths):
    #create output directory
    print(path)
    if not os.path.exists(output_dir[index]):
        os.makedirs(output_dir[index])
    if not os.path.exists(os.path.join(output_dir[index], model_name)):
        os.makedirs(os.path.join(output_dir[index], model_name))
        
    #iterate through subdirectories of given database path
    for sub in os.listdir(path):
        out_dir = os.path.join(output_dir[index], model_name, sub)
        #create subdirectory of output directory
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        sub_dir = os.path.join(path, sub)
        #extract feature for each file in subdirectory
        for root, dirs, files in os.walk(sub_dir):      
            for file in files:
                ext = file[-4:]
                img = cv2.imread(os.path.join(sub_dir, file)) 
                '''
                if(args.test_dataset=="mfr2"):
                   new_width=144
		   new_height=144
                   left = int((img.shape[0] - new_width)/2)
                   top = int((img.shape[1] - new_height)/2)
                   img = img[left:left+new_width, top:top+new_height,:]
                   img = cv2.resize(img, (112,112), interpolation=cv2.INTER_LINEAR)
                   #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                '''
                #img=np.minimum(np.maximum(img, 0.0), 255.0)
    
                nimg =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                aligned = np.transpose(nimg, (2,0,1))
                feature = model.get_feature(aligned)
                #print(feature.shape)
                #save feature
                save_file=os.path.join(out_dir, file.replace(ext, '.npy'))
                np.save(save_file, feature)
