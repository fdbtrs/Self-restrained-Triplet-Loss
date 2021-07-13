import random
import cv2
import dlib
import numpy as np
import os
from mtcnn import MTCNN

class FaceMasked:
    """
    Add simulated masked to face image
    """
    def __init__(self, facePredictor):
        """
        Instantiate an 'FaceMasked' object.
        facePredictor: The path to dlib's
        """
        assert facePredictor is not None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(facePredictor)
        self.mtcnndetector = MTCNN()

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

    def getboundingbox(self, img, multiFace=False):
        """"
        get the boundingbox
        :param img
        :param multiFace ignore the image if there are multi-faces
        """
        faces = self.getAllFaceBoundingBoxes(img)
        if (not multiFace and len(faces) > 0) or len(faces) == 1:
            return max(faces, key=lambda rect: rect.width() * rect.height())
        else:
            return None

    def findLandmarks(self, rgbImg, bb):
        assert rgbImg is not None
        assert bb is not None

        points = self.predictor(rgbImg, bb)
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

        # Line thickness of 1 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(image, str(i+1), org, font,
                            fontScale, color, thickness, cv2.LINE_AA)
      return  image
    def simulateMask(self, rgbImg=None, mask_type=None,color=None,draw_landmarks=False,  boundingbox=None, skipMulti=False, landmarks=None):
        rgbImg=np.array(rgbImg, dtype=np.uint8)
        if(mask_type==None):
            mask_type=np.random.choice(['a','b','c','d','e','f'])
        if boundingbox is None:
            boundingbox = self.getboundingbox(rgbImg, skipMulti)
            if boundingbox is None:
                boundingbox=self.mtcnndetector.detect_faces(img)['box']
                if (boundingbox is None):
                 return

        if landmarks is None:
            landmarks = self.findLandmarks(rgbImg, boundingbox)
        #npLandmarks = np.float32(landmarks)
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

        return result


if __name__ == '__main__':
    dl = FaceMasked('shape_predictor_68_face_landmarks.dat')
    list_images=os.listdir("loose_crop")
    list_masked=os.listdir('loose_crop_mask')
    ftx=open("ftxmtcnn.txt","w")
    for i in list_images:
      if not (i in list_masked):
        img_name=os.path.join("loose_crop",i)
        img = cv2.imread(img_name)
        img = dl.simulateMask(np.array(img, dtype=np.uint8), mask_type="a", color=(0, 0, 0), draw_landmarks=False)
        if not (img is None):
         cv2.imwrite(os.path.join("loose_crop_mask_mtcnn",i),img)
        else:
            ftx.write(i + "\n")


    #cv2.imshow("simulated", im)
    #cv2.waitKey()







