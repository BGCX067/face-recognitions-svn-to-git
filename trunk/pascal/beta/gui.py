# -*- coding: utf-8 -*-
"""
PyQt-GUI-Modul, Benutzeroberflaeche der gesamten Anwendung sowie noetige
Video-Bild-Konvertierungen für PyQt Support.
 
"""
import logging as log
from PyQt4 import Qt, QtCore, QtGui
import numpy as np
import cv2

import controller as c
import database as db
import fdetection as fd


class Video():
    """Klasse zum konvertieren des Videobilds und bereitstellung des ggf. bearbeiteten Frames"""
    def __init__(self, webcam, face_id=None, face_name=None, save_face=False, recognize_face=False, recognize_face_stopped=False):
        self.webcam = webcam
        self.face_id = face_id
        self.face_name = face_name
        self.save_face = save_face
        self.recognize_face = recognize_face
        self.stop = recognize_face_stopped
        self.current_frame=np.ndarray([])
        self.controller = c.Controller()
        
        self.controller.register_observer(self)
        self.observer = []
     
    # Observer-Pattern
    def register_observer(self, obj):
        """Wird zum registrieren eines Observers verwendet"""
        log.debug('registriere %s an Video', obj)
        self.observer.append(obj)
    def notify_observer(self):
        """Ruft update() aller zu benachritigen Objekte auf"""
        for obj in self.observer:
            obj.update()
    def update(self):
        """Wird vom Observierten Objekt aufgerufen wenn es sich geandert hat"""
        self.notify_observer()

    def capture_next_frame(self):
        """Liest naechsten Frame der Kamera und wandelt es von BGR zu RGB und startet die Gesichtserkennung"""
        success, read_frame=self.webcam.read()
        if success:              
            read_frame = cv2.cvtColor(read_frame, cv2.COLOR_BGR2RGB)
            self.current_frame = self.controller.frame_to_face(read_frame,self.face_id, self.face_name,
                                                               self.save_face, self.recognize_face)            
    
    def convert_frame(self):
        """Konvertiert Bild in ein von QtGUI akzeptiertes Format"""
        try:            
            height, width = self.current_frame.shape[:2]
            image = QtGui.QImage(self.current_frame, width, height,
                                 QtGui.QImage.Format_RGB888)
            image = QtGui.QPixmap.fromImage(image)
            #self.previous_frame = self.current_frame
            return image
        except:
            log.exception("Fehler beim konvertieren des Kamerabildes")
        
class GUI(QtGui.QMainWindow):
    """PyQt GUI fuer Button-Support und effiziente Kameraansteuerung."""
    def __init__(self, *args):
        """Buttons und ein Label fuer das Videobild sowie ein Timer zum 
        periodischen Ausfuehren der play() Methode
        
        """
        self.BUTTON_HEIGHT_DEFAULT = 30
        self.BUTTON_HEIGHT_BIG = 50 
        self.database = db.TrainingSets()
        #self.fdetection = fd.FaceDetector()
        
        # Hauptlayout Vertikal-Boxlayout
        QtGui.QWidget.__init__(self, *args)
        # selbst als Vater und Hauptwidget setzen 
        widget = QtGui.QWidget(self)
        self.setCentralWidget(widget)
        # vertikales Layout setzen
        v_parent_layout = QtGui.QVBoxLayout()    
        widget.setLayout(v_parent_layout)
        
        # QLabel als Videoframe Container
        self.video_label = QtGui.QLabel("Videobild")
        v_parent_layout.addWidget(self.video_label)
        
        # Bedienelemente
        palette = QtGui.QPalette()
        # Ausgabe Textfeld
        self.text_output = QtGui.QLineEdit(u"Pfüet Di! Wenn Du mich zum ersten mal startest mach mir bitte Dein Gesicht bekannt.", self)
        Qt.QObject.connect(self.text_output, Qt.SIGNAL('textChanged(const QString&)'), self.on_input_name)
        self.text_output.setMinimumHeight(40)
        v_parent_layout.addWidget(self.text_output)
        # Wer-Bin-Ich-Button
        self.button_who_i_am = QtGui.QPushButton("Wer-Bin-Ich?", self)
        self.button_who_i_am.setCheckable(True)
        palette.setColor(self.button_who_i_am.foregroundRole(), Qt.QColor("green"))
#         self.font_bold = QtGui.QFont("Arial", 55, QtGui.QFont.Bold)
#         self.button_who_i_am.fontChange(self.font_bold)
        self.button_who_i_am.setPalette(palette)
        Qt.QObject.connect(self.button_who_i_am, Qt.SIGNAL('clicked()'), self.clicked_who_i_am)
        self.button_who_i_am.setMinimumHeight(self.BUTTON_HEIGHT_BIG)
        v_parent_layout.addWidget(self.button_who_i_am)
        
        # SubLayout fuer Textfelder nebeneinander
        h_line_layout_text = QtGui.QHBoxLayout()
        v_parent_layout.addLayout(h_line_layout_text)
        # Namensfeld Textfeld
        self.text_name = QtGui.QLineEdit("Name", self)
        Qt.QObject.connect(self.text_name, Qt.SIGNAL('textChanged(const QString&)'), self.on_input_name)
        h_line_layout_text.addWidget(self.text_name)
        # ID Textfeld
        self.text_id = QtGui.QLineEdit("ID", self)
        Qt.QObject.connect(self.text_id, Qt.SIGNAL('textChanged(const QString&)'), self.on_input_id)
        h_line_layout_text.addWidget(self.text_id)
        # Training-Set-Aufnehmen-Button
        self.button_do_train = QtGui.QPushButton("Bekannt machen", self)
        self.button_do_train.setCheckable(True)
        self.button_do_train.setMinimumHeight(self.BUTTON_HEIGHT_DEFAULT)
        v_parent_layout.addWidget(self.button_do_train)
        Qt.QObject.connect(self.button_do_train, Qt.SIGNAL('clicked()'), self.clicked_do_train)
        # Beenden-Button
        self.button_quit = QtGui.QPushButton("Ende", self)
        v_parent_layout.addWidget(self.button_quit)
        palette.setColor(self.button_quit.foregroundRole(),Qt.QColor("red"))
        self.button_quit.setPalette(palette)
        Qt.QObject.connect(self.button_quit, Qt.SIGNAL('clicked()'), Qt.qApp,
                           Qt.SLOT('quit()'))
        Qt.QObject.connect(self.button_quit, Qt.SIGNAL('clicked()'), self.on_close)
                           
        self.button_quit.setMinimumHeight(self.BUTTON_HEIGHT_DEFAULT)
        v_parent_layout.addWidget(self.button_quit)

        # Observer registrieren, Video-Bild umwandeln und updaten
        self.webcam = cv2.VideoCapture(0)
        if self.webcam.isOpened(): 
            self.video = Video(self.webcam)
            self.video.controller.register_observer(self)
            self._timer = QtCore.QTimer(self)
            # timeout-SIGNAL an play-SLOT binden
            self._timer.timeout.connect(self.play)
            self._timer.start(27)
        else:            
            log.critical("Die Anwendung laeuft noch oder Web-Cam nicht angeschlossen!?")
    
    # Observer Pattern
    def update(self):
        """Wird vom Observierten Objekt aufgerufen wenn es sich geandert hat"""
        # Schreibt Text in Output Zeile der GUI
        self.text_output.setText(self.video.controller.info_text)
        
    def on_input_output(self, text):
        """Wird automatisch bei Eingabe ins Output-Textfeld aufgerufen. Sollte Leer bleiben!"""
        pass
    def on_input_name(self, text):
        """Wird automatisch bei Eingabe ins Name-Textfeld aufgerufen"""
        pass
    def on_input_id(self, text):
        """Wird automatisch bei Eingabe ins Textfeld aufgerufen"""
        pass
        
    def play(self):
        """Zum stetig wiederholtem Aufrufen um Kamerabild zu aktualisieren"""
        try:
            # naechsten Frame holen und diesen konvertiert als Pixmap in GUI anzeigen
            self.video.capture_next_frame()            
            self.video_label.setPixmap(self.video.convert_frame())            
        except Exception:
            log.exception("GUI.play(): Kein Bild von Kamera oder Konvertierungsproblem beim Wandeln zum QPixmap!")
    
    # Button-Callback-Funktionen
    def clicked_do_train(self):
        log.info('Training-Set: %s', self.button_do_train.isChecked())
        # Abfangen dass Facedetection Button gleichzeitig gedrueckt ist
        if self.button_who_i_am.isChecked():
            self.button_do_train.setChecked(False)
            log.error('Bitte erst Gesichtserkennung beenden!')
            return
        
        if self.button_do_train.isChecked():            
            self.button_do_train.setText("Anhalten")
            self.video.save_face = True            
            self.video.face_id = self.text_id.text()
            self.video.face_name = self.text_name.text()
            
            #self.video.zaheler = 1
            
        else: # not button.isChecked()
            #a = self.fdetection.get_speichern_ok()
            #print "fdetection Trainingsset-Bilder",a 
            
            self.button_do_train.setText("Training-Set")
            self.video.save_face = False
            
    def on_close(self):
        """Bei beenden des Programms aufrufen"""
        log.info('GUI on_close()')
        self.video.controller.on_close()
          
    def clicked_who_i_am(self):
        log.info('wer bin ich geklickt: %s', self.button_who_i_am.isChecked())
        # Verhindern dass gleichzeitig 'Train' Button aktiv ist
        if self.button_do_train.isChecked():                            
            self.button_who_i_am.setChecked(False)
            log.error('Bitte erst Trainings-Modus beenden!')
            return

        if not self.database.bilder_is_empty():
            log.info("Komm herein")

            if self.button_who_i_am.isChecked():
                self.button_who_i_am.setText("Anhalten")
                self.video.recognize_face = True
                self.video.face_id = self.text_id.text()

            else: # not button_who_i_am.isChecked()
                self.button_who_i_am.setText("Wer-Bin-Ich?")
                self.video.recognize_face = False
                self.video.stop = True
        else: # database.bilder_is_empty()            
            self.button_who_i_am.setChecked(False)
            button_msg = QtGui.QMessageBox(icon=QtGui.QMessageBox.Warning)
            button_msg.setText(u"Aktuell kenne ich noch kein Gesicht\n\n"
                               "Bitte mach uns zuerst bekannt, damit ich Dich wieder erkennen kann.")
            if button_msg.exec_() == QtGui.QMessageBox.Abort:
                return
    