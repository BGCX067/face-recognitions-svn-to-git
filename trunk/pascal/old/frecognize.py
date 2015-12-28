'''
Created on 27.01.2014

@author: jjoch001
'''
import model as m
import numpy as np
import cv2

class FaceRecognizer(object):
    '''
    classdocs
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.ts = m.TrainingSets()
    
    def face_recognize(self):
        [self.face_images, self.face_ids] = self.ts.get_faces()
        self.face_ids = np.asarray(self.face_ids, dtype=np.int32)
        #self.face_recog = cv2.createFisherFaceRecognizer()
        #self.face_recog.train(np.asarray(self.face_images), np.asarray(self.face_ids))
        #[predict_label, predict_confidence] = self.face_recog.predict(self.current_face))
        #print "Predicted Label =",predict_label,"(confidence=",predict_confidence,")"