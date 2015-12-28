# -*- coding: utf-8 -*-
"""
Modul um GUI Eingaben korrekt zu verarbeiten und entsprechende Prozeduren anzustossen.

"""
import os
import logging as log
try:
    import cPickle as pickle
except ImportError:
    import pickle    
from pprint import pprint as pp

from fdetection import FaceDetector as fd
import frecognize as fr
import database as db

class Controller(object):
    """Steuert Facedetector und Facerecognizer Objekte je nach Eingaben in der GUI."""

    def __init__(self):
        """Instanziiert immer ein Facedetector- und ein FaceRecognizer-Objekt.
        id_infos_dict {id = {'self.t_sets.KEY_NAME'='Pascal', self.t_sets.KEY_COUNT=0}, ... }
        TrainingSets besitzt die Keys-des Dictionaries als Konstanten.
        
        """
        self.NAME_SAVE_FILE = '.save.p'
        # dictionary mit Informationen zu den Personen
        self.id_infos_dict = {}
        self.t_sets = db.TrainingSets()
        self.save_path = self.t_sets.path
        self.sf = os.path.join(self.save_path,self.NAME_SAVE_FILE)
        try:
            self.id_infos_dict = pickle.load(open(self.sf, "rb" ))
            pp('unpickle %s' % self.id_infos_dict)
            log.debug('Aus Sicherungsdatei gelesenes Dictionary: %s', self.id_infos_dict)      
        except IOError as e:
            log.info('Die Sicherungsdatei wurde nicht gefunden. Beim ersten Programmstart korrekt.: %s', self.sf)
        except:
            log.exception('Unerwarteter Fehler beim lesen der Sicherungsdatei: %s', self.sf)
        self.id_infos_dict = self.t_sets.get_id_infos_dict(self.id_infos_dict)
        # Facedetekor-Objekt
        self.detect = fd()                
        self.fr = fr.FaceRecognizer()
        try:
            [face_images, face_ids] = self.t_sets.get_all_faces()
        except:
            log.exception("Es konnten keine Bilder geladen werden")
            raise
        if len(face_images) != 0:
            self.fr.trainFisherFaces(face_ids, face_images)
        else:
            log.info("Training Set ist leer oder Bilder können nicht geladen werden")
        self.trigger_rec = False
        self.trigger_save = False
        self.observer = []
        # Text der in Info-Zeile der GUI erscheint
        self._info_text = ''
    
    # Observer Pattern        
    def register_observer(self, obj):
        """Wird zum registrieren eines Observers verwendet"""
        log.debug('registriere %s an Controller', obj)
        self.observer.append(obj)
    def notify_observer(self):
        """Ruft update() aller zu benachritigen Objekte auf"""
        for obj in self.observer:
            obj.update()
    def get_percentage(self, total, part):
        """Berrechnung des Prozentualen Anteils von part an total."""
        percent = 0.0
        try:
            percent = 100 * part/float(total)
        except ZeroDivisionError:
            log.info('Es wurde in diesem Durchlauf KEIN Gesicht Erkannt!\nEventuell den Schwellwert ueberpruefen. total = %s', total)
        except:
            log.exception('Unerwarteter Fehler beim Berrechnen des Prozentanteils.')
        return percent
    
    def print_stat(self):
        """Erkennungs-Statistik-Ausgabe auf Konsole"""
        try:
            total = sum([v[self.t_sets.KEY_COUNT] for v in self.id_infos_dict.values()])
        except:
            log.exception('Fehler beim zusammenrechnen der Gesamtzahl moeglicher predicts. Totalsumme: %s\ninfo_dict', total, self.id_infos_dict)
        # Info-Text in GUI
        win_dic = (sorted(self.id_infos_dict.values(), key=lambda d: d[self.t_sets.KEY_COUNT], reverse=True)[0])
        name, face_id, percent = win_dic[self.t_sets.KEY_NAME], win_dic[self.t_sets.KEY_ID], self.get_percentage(total, win_dic[self.t_sets.KEY_COUNT])
        self.info_text = 'Du bist %s mit ID: %s und da bin ich zu %s%% sicher =))' % (name, face_id, percent)
        self.notify_observer()
        # Konsolen Ausgabe
        s = ['\n' + '-' * 40]
        for k, v in sorted(self.id_infos_dict.items(), key=lambda(k, v): v[self.t_sets.KEY_COUNT], reverse=True):  
            count = v[self.t_sets.KEY_COUNT]
            percentage = self.get_percentage(total, count)
            s.append('Predicts: ID: %s    %sx => %s%%' % (k, count, percentage))
        s.append('total: %s' %total)
        s.append('-' * 40 + '\n')
        log.info('\n'.join(s))
        
    def stopped_face_recognition(self):
        """Nach beenden des Facerecognition Modus"""
        # nur einmal bei Beenden der Gesichtserkennung
        log.info('Beende Gesichtserkennung...')
        self.trigger_rec = False
        self.print_stat()
        # Leeren der gemerkten predicts, damit bei nochmaligem Start die liste Leer ist
        for user in self.id_infos_dict.values():
            user[self.t_sets.KEY_COUNT] = 0
            
    def do_recognize_face(self):
        """Wird ausgefuehrt wenn Gesichtswiedererkennung Aktiviert wurde"""
        self.trigger_rec = True
        # Facedetection
        similar = self.fr.get_confidence(self.face)
        if similar < 0.7:
            predicted_face = self.fr.predict(self.face)
            confidence = (1.0 - min(max(similar,0.0),1.0)) *100
            # nur IDs ab 0 sind gueltig
            if predicted_face >= 0:
                try:                    
                    self.info_text = 'Hallo %s! ID-%s confidence: %s %% =)' % (self.id_infos_dict[str(predicted_face)][self.t_sets.KEY_NAME],
                                                            predicted_face, confidence)
                    self.notify_observer()
                    self.id_infos_dict[str(predicted_face)][self.t_sets.KEY_COUNT] += 1
                except:
                    log.exception('Fehler beim Zugriff auf das Info-Dictionary, auf ID: %s\n'
                                  'Der Key koennte falsch sein oder nicht existieren.\ninfo_dict: %s', str(predicted_face),
                                  self.id_infos_dict)
            else:
                self.info_text = 'Hallo unbekannte Person!'
        else:
            self.info_text = 'Hallo unbekannter Person!'
        self.notify_observer()
 
        
    # Diese Methode schlank halten, da sie pro Frame aufgerufen wird!        
    def frame_to_face(self, frame, face_id, face_name, save_face, recognize_face):
        """Verarbeitet pro Frame die Informationen der gedrueckten Buttons und gibt bearbeiteten Frame zurueck."""
        self.frame, self.face = self.detect.detectFace(frame)
        if self.face is not None:
            if save_face:
                # Training-Set erstellung
                self.trigger_save = True
                self.detect.acceptNewFace(self.face, face_id, face_name)
                self.info_text = '%s Bilder der ID: %s gespeichert.' % (self.detect.getCounter(), face_id)
                self.notify_observer()
            elif self.trigger_save:
                # Nur einmal nach Beenden der Training-Set Aufnahme
                self.trigger_save = False                
                log.info('Beende Training-Set...')
                log.info('Trainiere Fisher Faces mit neue Gesichter...')
                # eingegebenen Name und ID aus GUI im Dictionary speichern
                self.id_infos_dict[str(face_id)] = {self.t_sets.KEY_NAME : str(face_name),
                                                    self.t_sets.KEY_COUNT : 0,
                                                    self.t_sets.KEY_ID : str(face_id)}
                # Alle Bilder neu einlesen, da neue hinzugekommen sind
                [face_images, face_ids]=self.t_sets.get_all_faces()
                self.detect.setCounter(0)
                if len(face_images)!=0:
                    self.fr.trainFisherFaces(face_ids, face_images)
                # TODO: Lernen der neu aufgenommenen Bilder hier starten
            elif recognize_face:
                self.do_recognize_face()
            elif self.trigger_rec:
                self.stopped_face_recognition()                                   
        return self.frame
    
    def on_close(self):
        """Wird beim beenden des Programms aufgerufen"""
        log.info('controller on_close()')
        try:
            print 'zu picklendes dic ', self.id_infos_dict
            pickle.dump(self.id_infos_dict, open(self.sf, "wb" ))
        except:
            log.exception('Fehler beim Serialisieren des Controllers.')
        
    # Properties fuer info_text
    def get_info_text(self):
        return self._info_text
    def set_info_text(self, value):
        self._info_text = value
    def del_info_text(self):
        del self._info_text
    info_text = property(get_info_text, set_info_text, del_info_text, "Info-Text der in GUI-Ausgabezeile erscheint.")