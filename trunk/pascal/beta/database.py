# -*- coding: utf-8 -*-
"""
Modul zur Verwaltung der Persistenten Daten.
Die Klasse TrainingSets stelt Werkzeug-Methoden bereit, sie ist kein Singleton und kann ueberall wo sie benoetigt wird neu instanziiert werden.

"""
import os
import errno
import sys
import datetime
import logging as log

import cv2, numpy as np

class TrainingSets(object):
    """Klasse die IO-Methoden bereit stellt fuer die Training-Sets.
    Sie haelt selbst keine Daten und dient nur als Werkzeug.
     
    """
    def __init__(self, path=None):
        print 'instanziiere ts()'
        self.DELIMITER = '_'
        # Keys fuer das Dictionary
        self.KEY_SUM_IMGS = 'sum_imgs'
        self.KEY_ID = 'id'
        self.KEY_NAME = 'name'
        self.KEY_COUNT = 'count_predict'
        self.path = r'~/Dropbox/FACERECOGNITION/_TRAINING_SETS_' if path is None else path
        self.path = os.path.expanduser(self.path)
        self.extensions = ['.jpg', '.JPG', '.png', '.PNG']
        # infos_dict nur bei bedarf erstellen, da teuer mit vielen Dateizugriffen
        self.id_infos_dict = {} 
        self.init_folder_structure()
        
    def id_infos_dict_update(self, face_id, **dic):                              
        """Fuegt neue Key-Value Paare einre ID im id_infos_dict hinzu oder updated sie."""
        for k, v in dic.items():
            self.id_infos_dict.update(dic)
#             self.id_infos_dict[face_id][k] = v
    
    def get_id_infos_dict(self, dic = None):
        """Gibt Dictionary mit IDs als Key zurueck. 
        uebergebene Namen, werden von denen die hier von Platte gelesen werden ueberschrieben!

        return -> id_infos_dict {id = {'self.KEY_NAME'='Pascal', 'self.KEY_COUNT'=0, self.KEY_ID}, ... }
        
        """
        log.debug('get_id_infos_dict()...')
        join = os.path.join
        dic = {} if dic == None else dic
        assert(isinstance(dic, dict))
        try:
            # Alle ID-Ordner durchgehen und dem Dict hinzufuegen
            hdd = sorted([f for f in os.listdir(self.path) 
                          if os.path.isdir(join(self.path,f)) and f.isdigit()])
            # cleanup IDs die nur in  Sicherungdatei existieren 
            if dic:                
                for f_id in dic.keys():
                    if f_id not in hdd:
                        del dic[f_id]
            # IDs die nur auf Platte vorhanden sind oder auf Platte und Sicherungsdatei
            for f_id in hdd:
                # ID nur auf Platte
                if f_id not in dic.keys():
                    log.info('Die ID: %s ist noch nicht im dic, fuege hinzu...', f_id)                                        
                    dic[f_id] = {self.KEY_NAME : 'Alien', 
                                 self.KEY_COUNT : 0, 
                                 self.KEY_ID : f_id,
                                 self.KEY_SUM_IMGS : self.get_faces(f_id, [])[1] 
                                 }
                # ID auf Platte und aus Sicherungsdatei gelesen
                else:        
                    log.debug('Die ID: %s kenne ich auch schon aus Sicherungsdatei.', f_id)
                    print 'bisher %s' % (dic[f_id])
                    sum_img = self.get_faces(f_id, [])[1]
                    dic[f_id].update({self.KEY_ID:f_id, self.KEY_SUM_IMGS:sum_img})
                    print 'jetzt %s' % dic[f_id]
#                     dic[f_id].update(**{self.KEY_SUM_IMGS:self.get_faces(f_id, [])[1]})
#                     dic.update({self.KEY_ID:f_id, self.KEY_SUM_IMGS:self.get_faces(f_id,[])[1]})
            log.debug('Alle IDs von Platte: gelesen %s', map(int,hdd))
        except:
            log.exception('Fehler beim erstellen des Info-Dictionary anhand der Ordnernamen.')
        log.debug('das dict nach allen ops %s', dic)
        self.id_infos_dict = dic
        return dic

    def get_image_name(self, face_id, face_name):
        """Gibt den Bildnamen fuer ein neu zu speicherndes Gesicht zurueck.
        
        return -> ID_Name_Datum_Uhrzeit.png
        
        """
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        # Anfang muss ID_Name_ sein da dies in get_id_and_names() erwartet wird
        img_name = "%(id)s%(delim)s%(name)s%(delim)s%(time)s.png" % {'id':str(face_id), 
                                                              'delim':self.DELIMITER,
                                                              'name':face_name,
                                                              'time':now}
        return img_name
    
    def create_folder(self, path, face_id=''):
        """Legt einen neuen Ordner im Dateisystem an: path/name."""
        # check ob Ordner bereits existiert
        if not os.path.exists(path):
            log.info('%s wird angelegt...', path)
            try:
                os.makedirs(path)
            except OSError, e:
                if e.errno == errno.EEXIST:
                    log.debug('ignoriere bekanntes os.error errno.EEXIST: %s', e)
            except:
                log.exception('Unerwarteter Fehler beim Anlegen von %s', path)
    
    def init_folder_structure(self):
        """Legt Ordnerstrukur an"""
        self.create_folder(self.path)
    
    def get_id_dirs(self):
        """Gibt Verzeichnispfade aller aktuell vorhandenen IDs zurueck
        return -> ['path/toid1', 'path/toid2', ...]
        """
        dirs = []
        join = os.path.join
        try:
            for folder in [f for f in os.listdir(self.path) if os.path.isdir(join(self.path,f))]:
                dirs.append(join(self.path, folder))
        except:
            log.exception('Fehler beim Auslesen der ID-Verzeichnis-Pfade')
        log.debug('alle id dirs %s', dirs)
        return dirs
        
    def bilder_is_empty(self):
        """Return -> True wenn alle Ordner KEIN Bild enthalten"""
        log.debug('in bilder_is_empty()')
        join = os.path.join        
        if self.trainings_set_is_empty()==False:
            for folder in [f for f in os.listdir(self.path) if os.path.isdir(join(self.path,f))]:
                folder = join(self.path, folder)
                for dat in os.listdir(folder):
                    if dat[-4:] in self.extensions:
                        return False
        return True
    
    def trainings_set_is_empty(self):
        """Return -> True wenn der Wurzelordner KEIN Verzeichnis enthaelt"""
        log.debug('in trainingsset_is_empty()')     
        try:
            dirs = [d for d in self.get_id_dirs() if os.path.isdir(d)]
        except:
            log.exception('Fehler beim pruefen ob das Training-Set Wurzelverzeichnis leer ist.')
            raise
        is_empty = [] == dirs        
        return is_empty
            
    def save_face(self, face, face_id, face_name):                
        """Fuegt ein Gesichtsbild dem entsprechenden Ordner (self.ID) hinzu"""     
        assert(isinstance(face, np.ndarray))
        folder = os.path.join(self.path, str(face_id))
        full_path = os.path.join(folder,self.get_image_name(face_id, face_name))
        if not os.path.exists(folder):
            self.create_folder(self.path, face_id)
        try:
            cv2.imwrite(full_path, face)
            log.debug('Bild gespeichert: %s', full_path)
        except (IOError, Exception):
            log.exception("Fehler beim Abspeichern des Bildes: %s", full_path)
    
    def get_faces(self, face_id, face_images):
        """Liest alle Bilder einer bestimmten ID ein, face_images ist Liste in der die Bilder gespeichert werden.
        Dated automatisch die Instanz-Member self.id_infos_dict up.
        
        return -> ([face_images], num_imgs)
        
        face_images sind von Platte eingelesenen Bilder als cv2-numpy-Bildarrays
        num_imgs ist die Anzahl der erfolgreich von Platte eingelesenen Bilder.
        
        """
        join = os.path.join
        id_path = join(self.path, str(face_id))
        #face_id = id_path.split(os.sep)[-1]
        num_imgs = 0
        images = [f for f in os.listdir(id_path) 
                  if os.path.isfile(join(id_path, f)) and f[-4:] in self.extensions
                  ]
        for img in images:
            try:
                img_path = join(id_path, img)                         
                im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                face_images.append(np.asarray(im, dtype = np.uint8))
                num_imgs +=1
            except IOError as e:
                log.exception("I/O error{0}: {1}".format(e.errno, e.strerror))
                continue
            except TypeError as e:
                log.exception('Fehler beim einlesen der Datei %s.\n'
                              'Eventuell ist die Datei defekt.', img_path)
                continue
            except:
                log.exception("Nicht erwarteter Fehler beim einlesen der Datei "
                            "%s", img_path)
        log.info('ID %s: %s Bilder eingelesen', face_id, num_imgs)
        log.debug('num cv2.imread-imgs = %s = len(dateien) = %s' % (num_imgs, len(images)))
        
        return face_images, num_imgs
            
    def get_all_faces(self):
        """Einlesen der Gesichtsbilder von Platte mit zuordnung der jeweiligen ID durch 2 Listen.
        
        return -> ([face_images], [face_ids])
        
        """
        face_images, face_ids = [], []
        for dirname, dirnames, filenames in os.walk(self.path):
            for subdirname in sorted(dirnames):
                if subdirname.isdigit():
                    face_id = int(subdirname)                    
                    face_images, number = self.get_faces(face_id, face_images)
                    face_ids.extend([face_id] * number)
                else:
                    log.info('Ueberspringe den Ordner: %s da er keine gueltige ID darstellt.', subdirname)
        return face_images, face_ids
#     
# if __name__ == '__main__':
#     log.basicConfig(format='%(levelname)s: %(message)s', level=log.DEBUG)
#     ts = TrainingSets()
#     ts.get_id_and_names()
#     print 'ts empty ',ts.trainings_set_is_empty()
#     print 'bidler empty ', ts.bilder_is_empty()
# #     ts.trainings_set_is_empty()
    
