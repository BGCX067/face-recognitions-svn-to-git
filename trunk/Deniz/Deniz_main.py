import sys
import fdetection as fd
import cv2
import numpy as np
from PyQt4 import QtGui, QtCore


class Capture():
    def __init__(self):
        self.capturing = False
        self.webcam = cv2.VideoCapture(0)

    def startCapture(self):
        print "pressed start"
        self.capturing = True
        if self.webcam.isOpened(): # try to get the first frame
            self.test, self.frame = self.webcam.read()
            self.detect = fd.FaceDetector()    
        else:
            self.test = False
        
        while self.test:
            self.img = self.detect.detectFace(self.frame)
            cv2.imshow("preview", self.img)
            # get next frame
            self.test, self.frame = self.webcam.read()
            cv2.waitKey(5)
        cv2.destroyAllWindows()

    def checkCapture(self):
        print("detected the faces")

    def quitCapture(self):
        print "pressed Quit"
        #cap = self.c
        cv2.destroyAllWindows()
        self.webcam.release()
        QtCore.QCoreApplication.quit()
               
class Window(QtGui.QWidget):
    def __init__(self):

        QtGui.QWidget.__init__(self)
        self.setWindowTitle('Control Panel')

        self.capture = Capture()

        cv2.destroyWindow("preview")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        # Buttons erstellen
        self.start_button = QtGui.QPushButton('Start',self)
        self.start_button.clicked.connect(self.capture.startCapture)

        self.check_button = QtGui.QPushButton('Check\nFaces',self)
        self.check_button.clicked.connect(self.capture.checkCapture)

        self.quit_button = QtGui.QPushButton('Quit',self)
        self.quit_button.clicked.connect(self.capture.quitCapture)

        # Layout erstellen und Buttons hinzufuegen
        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.check_button)
        vbox.addWidget(self.quit_button)

        
        # Layout setzen
        self.setLayout(vbox)
        self.setGeometry(100,100,200,200)
        self.show()
        
if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())
