# -*- coding: utf-8 -*-

import cv2, math, numpy as np
import os
import haarcascades
from datetime import datetime
import model
class Haarcascades(object):    
    """Stellt die Haarcascade-XML Dateien bereit."""
    def __init__(self):
        assert(haarcascades.__file__)
        self.haarcascades_xml_path = os.path.split(haarcascades.__file__)[0]
        self.FRONTAL_FACE_DEFAULT = os.path.join(self.haarcascades_xml_path, 'haarcascade_frontalface_default.xml')
        self.FRONTAL_FACE_ALT2 = os.path.join(self.haarcascades_xml_path, 'haarcascade_frontalface_alt2.xml')
        self.PROFILE_FACE = os.path.join(self.haarcascades_xml_path, 'haarcascade_profileface.xml')
        self.EYE = os.path.join(self.haarcascades_xml_path, 'haarcascade_eye.xml')                          # 40%  5ms
        self.LEFT_EYE_2SPLITS = os.path.join(self.haarcascades_xml_path, 'haarcascade_lefteye_2splits.xml') # 60%  7ms
        self.MCS_LEFT_EYE = os.path.join(self.haarcascades_xml_path, 'haarcascade_mcs_lefteye.xml')         # 80% 18ms
        self.RIGHT_EYE_2SPLITS = os.path.join(self.haarcascades_xml_path, 'haarcascade_righteye_2splits.xml')
        self.MCS_MOUTH = os.path.join(self.haarcascades_xml_path, 'haarcascade_mcs_mouth.xml')
        self.MCS_NOSE = os.path.join(self.haarcascades_xml_path, 'haarcascade_mcs_nose.xml')
        
class FaceDetector(object):
    def __init__(self):        
        """Greift auf Haar-Cascade XML Dateien zu."""
        self.haarcascades = Haarcascades()
        self.lefteye_center = cv2.CascadeClassifier(self.haarcascades.LEFT_EYE_2SPLITS)
        self.righteye_center = cv2.CascadeClassifier(self.haarcascades.RIGHT_EYE_2SPLITS)
        self.classifier = cv2.CascadeClassifier(self.haarcascades.FRONTAL_FACE_ALT2)
        self.fpp = FacePreprocessor()
        self.training_set = model.TrainingSets()
        self.old_face = None
        self.old_time = datetime.now()
        
    def detectFace(self, frame,face_id,save_face):
        """Sucht nach Gesichtern und Augen im Frame und bei Erfolg wird das Gesichts-Preprocessing durchgefuehrt."""        
        self.img = frame.copy()
        self.face_id = face_id
        self.save_face = save_face
        assert(self.img.shape[2] == 3)
        # bereitet Bild fuer Gesichtserkennung vor
        # Konvertiert Bild zu ein Grauwertbild
        g_img = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        DETECTION_WIDTH = 320
        scale = self.img.shape[1] / float(DETECTION_WIDTH)
        # skaliert das Bild runter auf eine Breite von 320, falls bild groesser ist als 320
        if self.img.shape[1] > DETECTION_WIDTH:
            scaled_height = int(self.img.shape[0]/scale +0.5)
            small_g = cv2.resize(g_img, (DETECTION_WIDTH,scaled_height))
        else:
            small_g = g_img
        # Histrogramm Ausgleich wird noch angewendet
        small_g = cv2.equalizeHist(small_g)
        # TODO: checken ob mit mehr Parametern ggf. Performance erhoeht wird
        # Methode die nach Gesichter sucht, in diesem Fall nur das groesste
        faces = self.classifier.detectMultiScale(small_g,
                                                 flags=cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
                                                 )

        for (x,y,w,h) in faces:
            # Gesicht-Rechteck, Eigentschaften muessen wieder hochskaliert werden
            if self.img.shape[1] > DETECTION_WIDTH:
                x = int(x * scale + 0.5)
                y = int(y * scale + 0.5)
                w = int(w * scale + 0.5)
                h = int(h * scale + 0.5)
            # Gesicht Bereich wird eingezeichnet
            cv2.rectangle(self.img,(x,y),(x+w,y+h),(255,0,0),2)
            # TODO: sind diese zweischritte noetig?
            face = self.img[y:y+h, x:x+w]
            o_face = face.copy()
            # Gesicht wird an Methode detectEyes uebergeben um nach Augen zusuchen
            success, lefteye_center, righteye_center = self.detectEyes(face)
            # Nur wenn beide Augen gefunden werden, wird dieses Gesicht weiter an ein FacePreprocessor weiter gegeben
            if success:
                self.pp_face = self.fpp.doPreprocess(lefteye_center, righteye_center, o_face)
                #print self.save_face
                if self.save_face:
                    self.acceptNewFace(self.pp_face)
        return self.img
    
    def detectEyes(self, face):
        """Erkennung von rechtem und linken Auge und berrechnung der Augenmittelpunkte."""
        lefteye_center = [0,0]
        righteye_center= [0,0]
        # Festgelegte groessen wo sich typische weise die Augen sich im Gesicht befinden
        EYE_SX=0.12
        EYE_SY=0.17
        EYE_SW=0.37
        EYE_SH=0.36
        g_face = cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
        # Rechtes und linkes Augenbereich
        left_x=int(face.shape[0]*EYE_SX +0.5)
        top_y=int(face.shape[1]*EYE_SY +0.5)
        width_x=int(face.shape[0]*EYE_SW +0.5)
        height_y=int(face.shape[1]*EYE_SH +0.5)
        right_x=int(face.shape[0]*(1.0-EYE_SX-EYE_SW)+0.5)
        top_left_face = g_face[top_y:top_y+height_y,left_x:left_x+width_x]
        top_right_face = g_face[top_y:top_y+height_y,right_x:right_x+width_x]
        # Suchgebiete der Augen
        # TODO: brauchen wir die auskommentierten Zeilen?
        #cv2.rectangle(face,(left_x,top_y),(left_x+width_x,top_y+height_y),(0,255,255),2)
        #cv2.rectangle(face,(right_x,top_y),(right_x+width_x,top_y+height_y),(0,255,0),2)
        # In vorherfestgelegte Augenbereich werden die Augen individuell gesucht
        lefteye = self.lefteye_center.detectMultiScale(top_left_face)
        righteye = self.righteye_center.detectMultiScale(top_right_face)
        # Nur wenn je eine Augen sich in die Liste befindet werden die Bereiche eingezeichnet und Augenmittepunkt berechnet
        if len(lefteye)==1 and len(righteye)==1:
            for (ex,ey,ew,eh) in lefteye:
                cv2.rectangle(face[top_y:top_y+height_y,left_x:left_x+width_x],(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
                lefteye_center[0] = ex+(ew/2)+left_x
                lefteye_center[1] = ey + (eh/2)+top_y
            for (ex,ey,ew,eh) in righteye:
                cv2.rectangle(face[top_y:top_y+height_y,right_x:right_x+width_x],(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                righteye_center[0] = ex+(ew/2)+right_x
                righteye_center[1] = ey +(eh/2)+top_y
            return True, lefteye_center, righteye_center
        return False, lefteye_center, righteye_center
    
    def acceptNewFace(self, new_face):
        # TODO: Schauen ob Gesicht in 1 Sekunden Takt gemacht wurde und sich von vorherige unterscheidet
        current_time = datetime.now()
        simular = 1.0
        if self.old_face != None:
            simular = self.compare(new_face, self.old_face)
        result = current_time - self.old_time   
        if result.seconds >= 1 and simular >= 0.3:
            #new_face wird gespiegelt
            mirror_face = cv2.flip(new_face,1)
            self.training_set.save_face(new_face, self.face_id)
#             self.training_set.save_face(mirror_face, self.face_id)
            print "saved a new face"
            
#             cv2.namedWindow("Face")
#             cv2.imshow("Face", new_face)
#             cv2.namedWindow("Mirror")
#             cv2.imshow("Mirror", mirror_face)
            #TODO: new_face und mirror_face in Verzeichnis ID abspeichern, Label und Bildpfad in info.dat schreiben 
            self.old_time = current_time
            self.old_face = new_face.copy()
        
    def compare(self,new,old):
        l2 = cv2.norm(new,old,cv2.NORM_L2)
        return l2/(new.shape[0]*new.shape[1])
    
class FacePreprocessor(object):
    """Wichtige Bearbeitungsschritte um das Gesicht besser vergleichbar zu machen."""    
    def __init__(self):        
        self.FACE_WIDTH = 70
        self.FACE_HEIGHT = self.FACE_WIDTH
        
    def doPreprocess(self,lefteye_center, righteye_center, face):
        """Fuehrt Transformation, Histogrammausgleich, Weichzeichnungsfilter und Cropping des Gesichtes durch."""
        self.lefteye_center = lefteye_center
        self.righteye_center = righteye_center
        self.face = cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
        self.gTransform()
        self.allHistEqual()
        self.fpp_result = cv2.bilateralFilter(self.fpp_result,d=0,sigmaColor=20.0, sigmaSpace=2.0)
        self.ellipMask()
        return self.fpp_result
    
    def gTransform(self):
        """Transformiert das Gesicht, so dass die Augen horizontal ausgerichtet sind."""
        # Mittepunkt zw. die Augen
        eye_center = ((self.lefteye_center[0]+self.righteye_center[0])*0.5,(self.lefteye_center[1]+self.righteye_center[1])*0.5)
        # Winkel der Augen
        dx = (self.righteye_center[0] - self.lefteye_center[0])
        dy = (self.righteye_center[1] - self.lefteye_center[1])
        length = math.sqrt(dx*dx+dy*dy)
        angle = math.atan2(dy,dx)*180.0/math.pi
        # Konstante von der Auge, die man braucht
        LEFT_EYE_X = 0.16
        LEFT_EYE_Y = 0.14
        RIGHT_EYE_X = (1.0-0.16)
        desired_len = (RIGHT_EYE_X - LEFT_EYE_X)
        scale = desired_len * self.FACE_WIDTH/length
        # Rotations Matrix
        rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale)
        rot_mat[0][2] += (self.FACE_WIDTH * 0.5) - eye_center[0]
        rot_mat[1][2] += (self.FACE_HEIGHT * LEFT_EYE_Y) - eye_center[1]
        # Erst mit grauwerten definiert
        self.fpp_result = np.ndarray(shape=(self.FACE_HEIGHT,self.FACE_WIDTH), dtype=np.uint8)
        self.fpp_result[:,:] = 128
        self.fpp_result = cv2.warpAffine(self.face,rot_mat,self.fpp_result.shape)
        
    def allHistEqual(self):
        """Wendet einen partiellen Histogrammausgleich auf Gesicht an."""
        width = self.fpp_result.shape[1]
        # Histogramm Ausgleich wird auf getrennt auf linke und rechte Seite angewendet, mitte gemischt
        left = self.fpp_result[0:self.fpp_result.shape[0],0:width/2]
        right = self.fpp_result[0:self.fpp_result.shape[0],width/2:width]
        entire = cv2.equalizeHist(self.fpp_result)
        left = cv2.equalizeHist(left)
        right = cv2.equalizeHist(right)
        for x in range(width):
            for y in range(self.fpp_result.shape[0]):
                p = 0
                if x<(width/4):
                    p = left[y,x]
                elif x<(width/2):
                    l = left[y,x]
                    e = entire[y,x]
                    f = (x-width/4.0)/(width/4)
                    p = int((1.0-f) * l+ f*e +0.5)
                elif x < (width*3/4):
                    r = right[y,x-width/2]
                    e = entire[y,x]
                    f = (x-width/2.0)/(width/4)
                    p = int((1.0-f)*e+f*r+0.5)
                else:
                    p = right[y,x-width/2]
                self.fpp_result[y,x] = p
    
    def ellipMask(self):
        """Cropping mit ellyptischer Maske entfernt nicht relevante Gesichtsbereiche wie Wangen und Haare."""
        ellip= np.ndarray(shape=self.fpp_result.shape, dtype = np.uint8)
        ellip[:,:] = 0
        cv2.ellipse(ellip, 
                    (int(self.FACE_WIDTH*0.5+0.5),int(self.FACE_HEIGHT*0.4+0.5)),
                    (int(self.FACE_WIDTH*0.5+0.5),int(self.FACE_HEIGHT*0.8+0.5)),
                    0, 0,360, 255, cv2.cv.CV_FILLED)
        self.fpp_result[:,:]=np.where(ellip[:,:] == 0,0,self.fpp_result[:,:])
    
