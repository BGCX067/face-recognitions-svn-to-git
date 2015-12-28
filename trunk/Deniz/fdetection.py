#-*- coding: utf-8 -*-
__author__ = 'deniz'

import cv2, math, numpy as np

class FaceDetector(object):
    #greift auf Haar-Cascade XML Dateien
    def __init__(self):
        self.faceDefault = "haarcascade_frontalface_default.xml"
        self.faceAlt2 = "haarcascade_frontalface_alt2.xml"
        self.lefteyeCenter = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
        self.righteyeCenter = cv2.CascadeClassifier("haarcascade_righteye_2splits.xml")
        self.classifier = cv2.CascadeClassifier(self.faceAlt2)
    #Sucht nach Gesichter und Augen im frame und zeichnet die Bereiche ein     
    def detectFace(self, frame):
        self.img = frame.copy()
        # detect faces and draw bounding boxes
        assert(self.img.shape[2] == 3)
        g = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        DETECTION_WIDTH = 320
        scale = self.img.shape[1] / float(DETECTION_WIDTH)
        if self.img.shape[1] > DETECTION_WIDTH:
            scaled_height = int(self.img.shape[0]/scale +0.5)
            smallg = cv2.resize(g, (DETECTION_WIDTH,scaled_height))
        else:
            smallg = g
        smallg = cv2.equalizeHist(smallg)
        faces = self.classifier.detectMultiScale(smallg,
                                                 flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
                                                 )

        for (x,y,w,h) in faces:
            if self.img.shape[1] > DETECTION_WIDTH:
                x = int(x * scale + 0.5)
                y = int(y * scale + 0.5)
                w = int(w * scale + 0.5)
                h = int(h * scale + 0.5)
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            face = self.img[y:y+h, x:x+w]
            uFace = face.copy()
            success, lefteyeCenter, righteyeCenter = self.detectEyes(face)
            if success:
                pp = FacePreprocessor(lefteyeCenter,righteyeCenter,uFace)
                pp.doPreprocess()
        return self.img
    
    def detectEyes(self, face):
        lefteyeCenter = [0,0]
        righteyeCenter= [0,0]
        EYE_SX=0.12
        EYE_SY=0.17
        EYE_SW=0.37
        EYE_SH=0.36
        gface = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        #Rechtes und linkes Augenbereich
        lx=int(face.shape[0]*EYE_SX +0.5)
        ty=int(face.shape[1]*EYE_SY +0.5)
        wx=int(face.shape[0]*EYE_SW +0.5)
        hy=int(face.shape[1]*EYE_SH +0.5)
        rx=int(face.shape[0]*(1.0-EYE_SX-EYE_SW)+0.5)
        tlFace = gface[ty:ty+hy,lx:lx+wx]
        trFace = gface[ty:ty+hy,rx:rx+wx]
        #Suchgebiete der Augen
        #cv2.rectangle(face,(lx,ty),(lx+wx,ty+hy),(0,255,255),2)
        #cv2.rectangle(face,(rx,ty),(rx+wx,ty+hy),(0,255,0),2)
        lefteye = self.lefteyeCenter.detectMultiScale(tlFace)
        righteye = self.righteyeCenter.detectMultiScale(trFace)

        if len(lefteye)==1 and len(righteye)==1:
            for (ex,ey,ew,eh) in lefteye:
                cv2.rectangle(face[ty:ty+hy,lx:lx+wx],(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
                lefteyeCenter[0] = ex+(ew/2)+lx
                lefteyeCenter[1] = ey + (eh/2)+ty
            for (ex,ey,ew,eh) in righteye:
                cv2.rectangle(face[ty:ty+hy,rx:rx+wx],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                righteyeCenter[0] = ex+(ew/2)+rx
                righteyeCenter[1] = ey +(eh/2)+ty
            return True, lefteyeCenter, righteyeCenter
        return False, lefteyeCenter, righteyeCenter

class FacePreprocessor(object):

    def __init__(self, lefteyeCenter, righteyeCenter, face):
        self.lefteyeCenter = lefteyeCenter
        self.righteyeCenter = righteyeCenter
        self.face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        self.FACE_WIDTH = 70
        self.FACE_HEIGHT = self.FACE_WIDTH
        #self.face = face
    def doPreprocess(self):
        self.gTransform()
        self.allHistEqual()
        self.warped = cv2.bilateralFilter(self.warped,d=0,sigmaColor=20.0, sigmaSpace=2.0)
        self.ellipMask()
    def gTransform(self):
        #Mittepunkt zw. die Augen
        eyeCenter = ((self.lefteyeCenter[0]+self.righteyeCenter[0])*0.5,(self.lefteyeCenter[1]+self.righteyeCenter[1])*0.5)
        #Winkel der Augen
        dx = (self.righteyeCenter[0] - self.lefteyeCenter[0])
        dy = (self.righteyeCenter[1] - self.lefteyeCenter[1])
        length = math.sqrt(dx*dx+dy*dy)
        angle = math.atan2(dy,dx)*180.0/math.pi
        #Konstante von der Auge, die man braucht
        LEFT_EYE_X = 0.16
        LEFT_EYE_Y = 0.14
        RIGHT_EYE_X = (1.0-0.16)
        desiredLen = (RIGHT_EYE_X - LEFT_EYE_X)
        scale = desiredLen * self.FACE_WIDTH/length
        #Rotations Matrix
        rot_mat = cv2.getRotationMatrix2D(eyeCenter, angle, scale)
        rot_mat[0][2] += (self.FACE_WIDTH * 0.5) - eyeCenter[0]
        rot_mat[1][2] += (self.FACE_HEIGHT * LEFT_EYE_Y) - eyeCenter[1]
        #Erst mit grauwerten definiert
        self.warped = np.ndarray(shape=(self.FACE_HEIGHT,self.FACE_WIDTH), dtype=np.uint8)
        self.warped[:,:] = 128
        self.warped = cv2.warpAffine(self.face,rot_mat,self.warped.shape)
        
    def allHistEqual(self):
        width = self.warped.shape[1]
        #Histogramm Ausgleich wird auf getrennt auf linke und rechte Seite angewendet, mitte gemischt
        left = self.warped[0:self.warped.shape[0],0:width/2]
        right = self.warped[0:self.warped.shape[0],width/2:width]
        entire = cv2.equalizeHist(self.warped)
        left = cv2.equalizeHist(left)
        right = cv2.equalizeHist(right)
        for x in range(width):
            for y in range(self.warped.shape[0]):
                v = 0
                if x<(width/4):
                    v = left[y,x]
                elif x<(width/2):
                    l = left[y,x]
                    e = entire[y,x]
                    f = (x-width/4.0)/(width/4)
                    v = int((1.0-f) * l+ f*e +0.5)
                elif x < (width*3/4):
                    r = right[y,x-width/2]
                    e = entire[y,x]
                    f = (x-width/2.0)/(width/4)
                    v = int((1.0-f)*e+f*r+0.5)
                else:
                    v = right[y,x-width/2]
                self.warped[y,x] = v
        
        
    def ellipMask(self):
        ellip= np.ndarray(shape=self.warped.shape, dtype = np.uint8)
        ellip[:,:] = 0
        cv2.ellipse(ellip, 
                    (int(self.FACE_WIDTH*0.5+0.5),int(self.FACE_HEIGHT*0.4+0.5)),
                    (int(self.FACE_WIDTH*0.5+0.5),int(self.FACE_HEIGHT*0.8+0.5)),
                    0, 0,360, 255, cv2.cv.CV_FILLED)
        self.warped[:,:]=np.where(ellip[:,:] == 0,0,self.warped[:,:])
        cv2.namedWindow("Mask")
        cv2.imshow("Mask", self.warped)