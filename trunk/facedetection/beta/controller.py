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

from fdetection import FaceDetector as fd
import frecognize as fr
import database as db

class Controller(object):
    """Steuert Facedetector und Facerecognizer Objekte je nach Eingaben in der GUI."""

    def __init__(self):
        """Laed alten Zustand aus Sicherungsdatei, Instanziiert immer ein Facedetector- 
        und ein FaceRecognizer-Objekt
        
        """
        self.NAME_SAVE_FILE = '.save.p'
        # 3 Zustaende gut, normal, schlecht um z.b. in GUI Textfarbe entsprechend zu setzen
        self.__GOOD = 1
        self.__NORMAL = 0
        self.__BAD = -1
        # dictionary mit Informationen zu den Personen
        self.id_infos_dict = {}
        self.t_sets = db.TrainingSets()
        self.save_path = self.t_sets.path
        self.sf = os.path.join(self.save_path,self.NAME_SAVE_FILE)
        try:
            self.id_infos_dict = pickle.load(open(self.sf, "rb" ))
            log.debug('Aus Sicherungsdatei gelesenes Dictionary: %s', self.id_infos_dict)      
        except IOError as e:
            log.info('Die Sicherungsdatei wurde nicht gefunden. Beim ersten Programmstart korrekt.: %s', self.sf)
        except:
            log.exception('Unerwarteter Fehler beim lesen der Sicherungsdatei: %s', self.sf)
        # ID Infos aus Sicherungsdatei und von Platte zusammenfuehren
        self.id_infos_dict = self.t_sets.get_id_infos_dict(self.id_infos_dict)
        # Facedetekor-Objekt
        self.detect = fd()                
        self.fr = fr.FaceRecognizer()
        try:
            [face_images, face_ids] = self.t_sets.get_all_faces()
        except:
            log.exception("Fehler beim laden der Bilder geladen werden")
        if len(face_images) != 0:
            self.fr.trainFisherFaces(face_ids, face_images)
        else:
            log.info("Training Set ist leer, Okay beim ersten Programmstart")
        self.running_face_recognition = False
        self.running_save_face = False
        self.observer = []
        # Text der in Info-Zeile der GUI erscheint
        self.info_text = ''
        # -1, 0, 1 schlecht, normal, gut
        self.state = self.NORMAL
    
    # Observer Pattern        
    def register_observer(self, obj):
        """Wird zum registrieren eines Observers verwendet"""
        log.debug('registriere %s an Controller', obj)
        self.observer.append(obj)
        
    def notify_observer(self):
        """Ruft update() aller zu benachritigen Objekte auf"""
        for obj in self.observer:
            obj.update()
            
    def started_face_face_recognition(self):
        """Nur Einmal beim Starten der Training-Set-Aufnahme"""
        # Check ob genug Trainingsets vorhanden sind
        if len(self.id_infos_dict) > 2:
            too_little = []
            for k, v in self.id_infos_dict.items():
                sum_imgs = v[self.t_sets.KEY_SUM_IMGS]
                if sum_imgs < 101:
                    f_id, u_name, imgs = v[self.t_sets.KEY_NAME], k, sum_imgs
                    too_little.append((v[self.t_sets.KEY_NAME], 'id ' + str(k), 'sum: ' + str(sum_imgs)))
                    log.info('Zu wenig Bilder des Users: %s mit ID: %s und %s Bildern', f_id, u_name, imgs)
#             self.set_text_and_state(text)
            
    def started_save_face(self, face_id, face_name):
        """Nur Einmal beim Starten der Training-Set-Aufnahme"""
        face_id = str(face_id)
        face_name = str(face_name)
        # Check ob aktueller User neu ist
        if face_id in ['', 'ID']:
            text = 'ID nicht gueltig. Vergeben sind IDs: %s' % ','.join(self.id_infos_dict.keys())
            log.info(text)
            self.set_text_and_state(text, self.BAD)
            return False
        self.state = self.GOOD 
        if not self.id_infos_dict.has_key(face_id): #and self.id_infos_dict[face_id].has_key(self.t_sets.KEY_COUNT):
            self.id_infos_dict[face_id] = {self.t_sets.KEY_NAME : face_name,
                                               self.t_sets.KEY_COUNT : 0,  #  fuer stat
                                               self.t_sets.KEY_ID : face_id,
                                               self.t_sets.KEY_SUM_IMGS : 0}
            log.info('Neuer user angelegt...jetzt dic ', self.id_infos_dict)
        return True

    def do_save_face(self, face_id, face_name):
        """Wird pro Frame ausgefuehrt wenn Training-Set aufgenommen wird also der 'Bekannt-Machen-Button' aktiviert ist."""
        # PyQt Strings in Pythonstrings wandeln
        face_id = str(face_id)
        face_name = str(face_name)
        self.running_save_face = True
        # Test ob ID und UserDict bereits vorhanden
        if self.id_infos_dict.has_key(face_id) and self.id_infos_dict[face_id].has_key(self.t_sets.KEY_COUNT):
            log.debug('dict ist da ich zaehle versuche gesicht zu accepten ...')
            imgs_accepted = self.detect.acceptNewFace(self.face, face_id, face_name)
            if imgs_accepted:
                self.id_infos_dict[face_id][self.t_sets.KEY_SUM_IMGS] += imgs_accepted
                sum_imgs = self.id_infos_dict[face_id][self.t_sets.KEY_SUM_IMGS]
                info = '%s Bilder der ID: %s gespeichert.' % (sum_imgs, face_id)
                self.set_text_and_state(info, self.GOOD if sum_imgs > 99 else self.BAD)
#                 self.notify_observer()
                log.info(info)
        else:
            log.critical('ID-%s nicht vorhanden beim Versuch Fotos hinzuzufuegen, sollte in started_save_face() angelegt werden.', face_id)
  
    def do_face_recognition(self):        
        """Wird pro Frame ausgefuehrt wenn Gesichtswiedererkennung Aktiviert wurde"""
        self.running_face_recognition = True
        # Check ob genug TrainingSets vorhanden sind
        if len(self.id_infos_dict) >= 3:
            similar = self.fr.get_similar(self.face)
            if similar < 0.7:
                predicted_face = self.fr.predict(self.face)
                confidence = (1.0 - min(max(similar,0.0),1.0)) *100
                # nur IDs ab 0 sind gueltig
                if predicted_face >= 0:
                    try:                        
                        self.state = self.NORMAL
                        text = 'Hallo %s! ID-%s confidence: %s %% =)' % (self.id_infos_dict[str(predicted_face)][self.t_sets.KEY_NAME],
                                                                predicted_face, confidence)
                        self.set_text_and_state(text, self.NORMAL)
                        self.id_infos_dict[str(predicted_face)][self.t_sets.KEY_COUNT] += 1
                    except:
                        log.exception('Fehler beim Zugriff auf das Info-Dictionary, auf ID: %s\n'
                                      'Der Key koennte falsch sein oder nicht existieren.\ninfo_dict: %s', str(predicted_face),
                                      self.id_infos_dict)
                else:
                    self.set_text_and_state('Hallo unbekannte Person!', self.NORMAL)
            else:
                self.set_text_and_state('Hallo unbekannte Person!', self.NORMAL)            
    
    def stopped_save_face(self, face_id, face_name):
        """Nur einmal beim Beenden einer Bilderserie, neu eingegebene Daten merken und neu anlernen.
        Code der nach dem Beenden des Facerecognition Modus genau einmal ausgefuehrt wird.
        
        """
        self.running_save_face = False                
        log.info('Beende Training-Set...')        
        # User Dict updaten und zaehler zuruecksetzen
        self.id_infos_dict[str(face_id)].update({self.t_sets.KEY_NAME : str(face_name),
                                                 self.t_sets.KEY_COUNT : 0,  # zuruecksetzen fuer stat
                                                 self.t_sets.KEY_ID : str(face_id)})
        log.debug('Das Dict nach ende der TS Aufnahme %s ', self.id_infos_dict)
        # Alle Bilder neu einlesen, da neue hinzugekommen sind
        log.info('Lese neu hinzugekommene Bilder von Platte ein...')
        [face_images, face_ids]=self.t_sets.get_all_faces()
        # Counter neu hinzugefuegter Bilder zurueck setzen
        self.detect.setCounter(0)
        if len(face_images)!=0:
            log.info('Trainiere Fisher Faces mit neue Gesichter...')
            self.fr.trainFisherFaces(face_ids, face_images)
            self.notify_observer()
    
    def stopped_face_recognition(self):
        """Nur einmal, nach beenden des Facerecognition Modus"""
        # nur einmal bei Beenden der Gesichtserkennung
        log.info('Beende Gesichtserkennung...')
        self.running_face_recognition = False
        self.print_stat()
        # Leeren der gemerkten predicts, damit bei nochmaligem Start die liste Leer ist
        for user in self.id_infos_dict.values():
            user[self.t_sets.KEY_COUNT] = 0
         
    # Diese Methode schlank halten, da sie pro Frame aufgerufen wird!        
    def frame_to_face(self, frame, face_id, face_name, save_face, recognize_face):
        """Verarbeitet pro Frame die Informationen der gedrueckten Buttons und gibt bearbeiteten Frame zurueck."""
        self.frame, self.face = self.detect.detectFace(frame)
        if self.face is not None:
            # Training-Set erstellung
            if save_face:
                self.do_save_face(face_id, face_name)
                self.running_save_face = True                
            # Nur einmal nach Beenden der Training-Set Aufnahme
            elif self.running_save_face:
                self.stopped_save_face(face_id, face_name)
            # Facerecognition durchfuehren
            # Klick Wo_i_am_Button
            elif recognize_face:                
                self.do_face_recognition()
            # Wer-Bin-Ich-Button deaktiviert
            elif self.running_face_recognition:
                self.stopped_face_recognition()                                   
        return self.frame    
    
    def print_stat(self):
        """Erkennungs-Statistik-Ausgabe auf Konsole.
        
        return -> 'Du bist <Username> mit ID: <ID> und da bin ich zu <xx.xxx>% sicher =))'
        
        """
        try:
            total = sum([v[self.t_sets.KEY_COUNT] for v in self.id_infos_dict.values()])
        except:
            log.exception('Fehler beim zusammenrechnen der Gesamtzahl moeglicher predicts. Totalsumme: %s\ninfo_dict', total, self.id_infos_dict)
        # Info-Text in GUI
        win_dic = (sorted(self.id_infos_dict.values(), key=lambda d: d[self.t_sets.KEY_COUNT], reverse=True)[0])
        name, face_id, percent = win_dic[self.t_sets.KEY_NAME], win_dic[self.t_sets.KEY_ID], self.get_percentage(total, win_dic[self.t_sets.KEY_COUNT])
        winner_text = 'Du bist %s mit ID: %s und da bin ich zu %s%% sicher =))' % (name, face_id, percent)
        self.set_text_and_state(winner_text, self.GOOD if percent > float(50) else self.BAD)
        # Konsolen Ausgabe
        s = ['\n' + '-' * 40]
        for k, v in sorted(self.id_infos_dict.items(), key=lambda(k, v): v[self.t_sets.KEY_COUNT], reverse=True):  
            count = v[self.t_sets.KEY_COUNT]
            percentage = self.get_percentage(total, count)
            s.append('Predicts: ID: %s    %sx => %s%%' % (k, count, percentage))
        s.append('total: %s' %total)
        s.append('-' * 40 + '\n')
        text = '\n'.join(s)
        print text
        #log.info(text)
        return winner_text
    
    def on_close(self):
        """Wird beim beenden des Programms aufgerufen"""
        log.debug('controller on_close()')
        try:
            log.info('Sicherungsdatei speichern...')
            log.debug('Info-Dictionary das ich in Sicherungsdatei serialisiere %s', self.id_infos_dict)
            pickle.dump(self.id_infos_dict, open(self.sf, "wb" ))
        except:
            log.exception('Fehler beim Serialisieren des Controllers.')
    
    def set_text_and_state(self, text, state=None):
        """Setzt Info-Text und Status, benachrichtigt anschliessend die Observer"""
        if state:
            self.state = state
        self.info_text = text        
        self.notify_observer()
        
    def get_sum_of_users(self):
        return len(self.id_infos_dict)
    
    def get_percentage(self, total, part):
        """Berechnung des Prozentualen Anteils von part an total."""
        percent = 0.0
        try:
            percent = 100 * part/float(total)
        except ZeroDivisionError:
            log.info('Es wurde in diesem Durchlauf KEIN Gesicht Erkannt!\nEventuell den Schwellwert ueberpruefen. total = %s', total)
        except:
            log.exception('Unerwarteter Fehler beim Berrechnen des Prozentanteils.')
        return percent
    
    @property
    def GOOD(self):
        return self.__GOOD
    @property
    def NORMAL(self):
        return self.__NORMAL
    @property
    def BAD(self):
        return self.__BAD