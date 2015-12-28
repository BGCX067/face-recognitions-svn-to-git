'''
Verwaltung der Persistenten Daten und Programmzustaende.

'''
import os
import errno
import sys
import logging
import time

import cv2, numpy as np
import hashlib
import binascii 

class TrainingSets(object):
    """Verwaltung der TrainingSets (vorab gespeicherte Gesichtsbilder) der Personen und ihrer IDs."""
    
    def __init__(self, path='~/Dropbox/FACERECOGNITION/_TRAINING_SETS_'):
        self.path = os.path.expanduser(path)
        self.counter = 0
        self.images = {}
        print 'Wurzelpfad der TrainingSets: ', self.path
        # Logging
        self.log = time.strftime('%Y-%m-%d_%H-%M', time.gmtime()) + '.log'
#         logging.basicConfig(filename=self.log,level=logging.INFO)
        logging.basicConfig(filename=os.path.join(self.path,self.log), level=logging.INFO, format='%(asctime)s %(message)s')
        logging.info('START ------------------------------------------')
        print self.log
        self.init_folder_structure()
    
    def create_folder(self, path, face_id=''):
        """Legt einen neuen Ordner im Dateisystem an: path/name."""
        # check ob Ordner bereits existiert
        path = os.path.join(path, str(face_id))
        if not os.path.exists(path):
            print path , 'wird angelegt...'
            try:
                os.makedirs(path)
            except OSError, e:
                if e.errno == errno.EEXIST:
                    print 'ignoriere os.error'
        else:
            print path, ' bereits vorhannden'        
    
    # todo raus
    def init_folder_structure(self):
        """Legt Ordnerstrukur an"""
        self.create_folder(self.path)

    def save_face(self, face, face_id):                
        """Fuegt ein Gesichtsbild dem entsprechenden Ordner (self.ID) hinzu"""     
        assert(isinstance(face, np.ndarray))
        folder = os.path.join(self.path, str(face_id))
        if not os.path.exists(folder):
            self.create_folder(self.path, face_id)
        face = np.asarray(face, dtype = np.uint8)
        image_name = self.get_image_name(face_id, face)
        cv2.imwrite(os.path.join(folder, image_name), face)
        self.counter += 1
    
    def get_image_name(self, face_id, face):
        """Gibt den Bildnamen fuer ein neu zu speicherndes Gesicht zurueck"""
        assert(isinstance(face, np.ndarray))
        name = '%s_%s.jpg' % (str(face_id), self.get_hash(face.tostring())) #self.counter)
        return name

    def compare(self, hash, img):
        """Vergleicht einen Hashwert mit dem eines Bild"""
#         hash_before = img.split('_')[1][:-4]
        hash_after = self.get_hash(img.tostring())
        print hash, ' vorher'
        print hash_after, ' nachher'
        print '--> ', hash == hash_after
        
    def get_faces(self, id_path):
        """Liest alle Bilder einer bestimmten ID ein"""
        id_path = os.path.join(self.path, id_path)
        print 'get_faces()'
        print 'id_path ', id_path
        
        face_images = []
        for img in os.listdir(id_path):
            try:
                img_path = os.path.join(id_path, img)
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face_images.append(np.asarray(im, dtype = np.uint8))
                self.compare(img.split('_')[1][:-4], im)
            except IOError,(errno,strerror):
                print "I/O error{0}: {1}".format(errno, strerror)
            except:
                print "Unexpected error: ", sys.exc_info()[0]
                raise
        print 'Ich habe %s Bilder eingelesen' % len(face_images)
        return face_images, len(face_images)
            
    # warum nicht jeder face_id ne liste von face_images zuordnen?
    def get_all_faces(self):
        """Einlesen der Gesichtsbilder von Platte mit zuordnung der jeweiligen ID durch 2 Listen."""
        counter = 0
        face_images, face_ids = [], []
        for dirname, dirnames, filenames in os.walk(self.path):
            for subdirname in dirnames:
                if subdirname != 'ID':
                    logging.warning('--------------------')
                    logging.warning('subdirname '+ subdirname)
                    id_path = os.path.join(dirname, subdirname)
                    images, number = self.get_faces(id_path)
                    face_images.extend(images)
                    face_ids.extend([int(subdirname)] * number)
                    print face_ids
                    print number
        print len(face_images)
        return [face_images, face_ids]
    
        
    def get_hash(self, string):
        """Gibt md5 Hashwert des Objekts als String zurueck"""
        m = hashlib.md5()
        m.update(string)
        digest = binascii.hexlify(m.hexdigest())
        print digest
        return digest


        
