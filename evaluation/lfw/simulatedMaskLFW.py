import random

import cv2
import dlib
import numpy as np
import os


class AlignDlib:
    """
    Use `dlib's landmark estimation <http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html>`_ to align faces.
    The alignment preprocess faces for input into a neural network.
    Faces are resized to the same size (such as 96x96) and transformed
    to make landmarks (such as the eyes and nose) appear at the same
    location on every image.
    Normalized landmarks:
    .. image:: ../images/dlib-landmark-mean.png
    """

    #: Landmark indices corresponding to the inner eyes and bottom lip.
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]

    #: Landmark indices corresponding to the outer eyes and nose.
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, facePredictor):
        """
        Instantiate an 'AlignDlib' object.
        :param facePredictor: The path to dlib's
        :type facePredictor: str
        """
        assert facePredictor is not None

        #pylint: disable=no-member
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)

    def getAllFaceBoundingBoxes(self, rgbImg):
        """
        Find all face bounding boxes in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :return: All face bounding boxes in an image.
        :rtype: dlib.rectangles
        """
        assert rgbImg is not None

        try:
            return self.detector(rgbImg, 1)
        except Exception as e: #pylint: disable=broad-except
            print("Warning: {}".format(e))
            # In rare cases, exceptions are thrown.
            return []

    def getLargestFaceBoundingBox(self, rgbImg, skipMulti=False):
        """
        Find the largest face bounding box in an image.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param skipMulti: Skip image if more than one face detected.
        :type skipMulti: bool
        :return: The largest face bounding box in an image, or None.
        :rtype: dlib.rectangle
        """
        assert rgbImg is not None

        faces = self.getAllFaceBoundingBoxes(rgbImg)
        if (not skipMulti and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        """
        Find the landmarks of a face.
        :param rgbImg: RGB image to process. Shape: (height, width, 3)
        :type rgbImg: numpy.ndarray
        :param bb: Bounding box around the face to find landmarks for.
        :type bb: dlib.rectangle
        :return: Detected landmark locations.
        :rtype: list of (x,y) tuples
        """
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
        #return list(map(lambda p: (p.x, p.y), points.parts()))
        return [(p.x, p.y) for p in points.parts()]
    def put_text(self,image,landmark):
        # font
      for i in range(0,len(landmark)):
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (landmark[i][0],landmark[i][1])

        # fontScale
        fontScale = 0.4

        # Blue color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, str(i+1), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
      return  image
    def simulateMask(self, rgbImg=None, mask_type=None,color=None,draw_landmarks=False,  skipMulti=False, landmarks=None,boundingbox=None):
        rgbImg=np.array(rgbImg, dtype=np.uint8)
        if(mask_type==None):
            mask_type=np.random.choice(['a','b','c','d','e','f'])
        if boundingbox is None:
            bb = self.getLargestFaceBoundingBox(rgbImg, skipMulti)
            if bb is None:
                return
        else:
            bb = dlib.rectangle(int(float(boundingbox[0])), int(float(boundingbox[1])),
                                         int(float(boundingbox[2])), int(float(boundingbox[3])))
        print(boundingbox)
        result=None

        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, bb)

        npLandmarks = np.float32(landmarks)
        if(color is None):
            color=  (random.uniform(0, 255), random.uniform(0,255), random.uniform(0,255))
        if(mask_type=='c'):
            pts=np.array([[landmarks[1],landmarks[2],landmarks[3],landmarks[4],landmarks[5],landmarks[6],landmarks[7],
                      landmarks[8],landmarks[9],landmarks[10],landmarks[11],landmarks[12],landmarks[13],landmarks[14],landmarks[15],landmarks[29]]],dtype=np.int32)
            result=cv2.fillPoly(rgbImg,pts,color,lineType = 8)
        elif(mask_type=='a'):
            pts = np.array(
                [[landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7],
                  landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14],
                  landmarks[15],[landmarks[42][0],landmarks[15][1]],  landmarks[27],[landmarks[39][0],landmarks[1][1]] ]], dtype=np.int32)
            result=cv2.fillPoly(rgbImg,pts,color,lineType = 8)
        elif(mask_type=='b'):
            top=( landmarks[27][1] + (landmarks[28][1]-landmarks[27][1])/2)
            center=(landmarks[28][0], top+ (landmarks[8][1]-top)/2)
            axis_x=int((landmarks[13][0]-landmarks[3][0])*0.8)
            axis_y=landmarks[8][1]-top
            axis=(int(axis_x),int(axis_y))
            result=cv2.ellipse(rgbImg,(center,axis,0),color=color,thickness=-1,lineType=8)
        elif (mask_type=='d'):
            top = landmarks[29][1]
            center = (landmarks[28][0], top + (landmarks[8][1] - top) / 2)
            axis_x = int((landmarks[13][0] - landmarks[3][0]) * 0.8)
            axis_y = landmarks[8][1] - top
            axis = (int(axis_x), int(axis_y))
            result = cv2.ellipse(rgbImg, (center, axis, 0), color=color, thickness=-1, lineType=8)
        elif (mask_type == 'f'):
            iod=landmarks[46][0]-landmarks[40][0]
            top = landmarks[29][1]+0.33*iod
            center = (landmarks[28][0], top + (landmarks[8][1] - top) / 2)
            axis_x = int((landmarks[13][0] - landmarks[3][0]) * 0.8)
            axis_y = landmarks[8][1] - top
            axis = (int(axis_x), int(axis_y))
            result = cv2.ellipse(rgbImg, (center, axis, 0), color=color, thickness=-1, lineType=8)
        elif (mask_type == 'e'):
            pts = np.array(
                [[landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7],
                  landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14],
                  landmarks[15], landmarks[35], landmarks[34], landmarks[33], landmarks[32], landmarks[31]]], dtype=np.int32)
            result = cv2.fillPoly(rgbImg, pts, color, lineType=8)
        if(draw_landmarks):
         result=self.put_text(result,landmarks)
        if(result is None):
            print("Landmark could not be detected")
            print(result.shape)

        return result,mask_type,color

def read_pairs_probe(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            if(len(pair)>3):
                pairs.append(pair[2]+'_'+pair[3].zfill(4)+'.jpg')
            else:
                pairs.append(pair[0]+'_'+pair[2])

    return np.array(pairs)

if __name__ == '__main__':
    dl = AlignDlib('shape_predictor_68_face_landmarks.dat')

    source_folder='/home/aboller/ArcFace/Data/lfw'
    paris=read_pairs_probe('log/pairs.txt')

    for probr, probd, probfiles in os.walk(source_folder):
        for pfile in probfiles:
            #f(pfile in paris):
                img = cv2.imread(os.path.join(probr, pfile))
                im = dl.simulateMask(np.array(img, dtype=np.uint8),  draw_landmarks=False)
                cv2.imwrite(os.path.join(probr, pfile), im)
                print(pfile)






