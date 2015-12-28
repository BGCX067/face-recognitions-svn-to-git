# -*- coding: utf-8 -*-
'''
PyQt-GUI-Modul, Benutzeroberflaeche der gesamten Anwendung sowie noetige
Video-Bild-Konvertierungen für PyQt Support.
 

'''
from PyQt4 import Qt, QtCore, QtGui
import cv2
import sys

import fdetection as fd
import model
import numpy as np


class Video():
    """Klasse zum konvertieren des Videobilds"""
    def __init__(self, webcam, face_id=None, save_face=False):
        self.webcam = webcam
        self.face_id = face_id
        self.save_face = save_face
        self.current_frame=np.ndarray([])
        self.detect = fd.FaceDetector()
        
 
    def capture_next_frame(self):
        """Liest naechsten Frame der Kamera und wandelt es von BGR zu RGB und startet die Gesichtserkennung"""
        success, read_frame=self.webcam.read()
        if success:              
            read_frame = cv2.cvtColor(read_frame, cv2.COLOR_BGR2RGB)
            self.current_frame = self.detect.detectFace(read_frame, self.face_id,self.save_face)
            
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
            print "Fehler beim konvertieren des Kamerabildes: ", sys.exc_info()[0]
            raise
        
class GUI(QtGui.QMainWindow):
    """PyQt GUI fuer Button-Support und effiziente Kameraansteuerung."""
    
    def __init__(self, *args):
        """Buttons und ein Label fuer das Videobild sowie ein Timer zum 
        periodischen Ausfuehren der play() Methode
        
        """      
        QtGui.QWidget.__init__(self, *args)
        self.trainingSets = model.TrainingSets()
        
        # selbst als Vater und Hauptwidget setzen 
        widget = QtGui.QWidget(self)
        self.setCentralWidget(widget)
        
        # vertikales Layout setzen
        boxLayout = QtGui.QVBoxLayout()    
        widget.setLayout(boxLayout)
        
        # QLabel als Videoframe Container
        self.video_label = QtGui.QLabel("Videobild")
        boxLayout.addWidget(self.video_label)

        # ID Textfeld
        self.id_text = QtGui.QLineEdit("ID", self)
        Qt.QObject.connect(self.id_text, Qt.SIGNAL('textChanged(const QString&)'), self.do_training_set)
        boxLayout.addWidget(self.id_text)
        
        # Training-Set-Aufnehmen-Button
        self.training_set_button = QtGui.QPushButton("Training-Set", self)
        self.training_set_button.setCheckable(True)
        boxLayout.addWidget(self.training_set_button)
        Qt.QObject.connect(self.training_set_button, Qt.SIGNAL('clicked()'), self.training_set_clicked)
                        
        # Wer-Bin-Ich-Button
        self.who_am_i_button = QtGui.QPushButton("Wer-Bin-Ich?", self)
        boxLayout.addWidget(self.who_am_i_button)
        self.who_am_i_button.clicked.connect(self.who_i_clicked)
        
        # Beenden-Button
        self.quit_button = QtGui.QPushButton("Ende", self)
        boxLayout.addWidget(self.quit_button)
        palette = QtGui.QPalette()
        palette.setColor(self.quit_button.foregroundRole(),Qt.QColor("red"))
        self.quit_button.setPalette(palette)
        Qt.QObject.connect(self.quit_button, Qt.SIGNAL('clicked()'), Qt.qApp,
                           Qt.SLOT('quit()'))
        boxLayout.addWidget(self.quit_button)

        # Video-Bild umwandeln und updaten
        self.webcam = cv2.VideoCapture(0)
        if self.webcam.isOpened(): 
            self.video = Video(self.webcam)
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self.play)
            self._timer.start(27)
            self.update()
        else:
            self.test = False
            print "Web-Cam nicht angeschlossen"

    def do_training_set(self, text):
        print 'do_training_set(): ', text
        
    def play(self):
        """Zum stetig wiederholtem Aufrufen um Kamerabild zu aktualisieren"""
        try:
            # naechsten Frame holen und diesen konvertiert als Pixmap in GUI anzeigen
            self.video.capture_next_frame()            
            self.video_label.setPixmap(self.video.convert_frame())            
        except TypeError:
            print "GUI.play(): Kein Bild von Kamera oder Bild-Konvertierungsproblem!"
    
    # Button-Callback-Funktionen
    def training_set_clicked(self):
        print 'training_set_clicked() und button-status: ', self.training_set_button.isChecked()
        if self.training_set_button.isChecked():
            self.training_set_button.setText("Anhalten")
            self.video.save_face = True
            self.video.face_id = self.id_text.text()
        else: # not button.isChecked()
            self.training_set_button.setText("Training-Set")
            self.video.save_face = False
        
    def who_i_clicked(self):
        print "Who i am"
        self.trainingSets.get_faces('/home/mi/ptreb001/Dropbox/FACERECOGNITION/_TRAINING_SETS_/ID')
#         print len(self.trainingSets.get_all_faces()), ' len'
        