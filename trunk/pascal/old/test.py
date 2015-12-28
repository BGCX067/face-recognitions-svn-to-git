'''
Created on 29.01.2014

@author: ptreb001
'''
import unittest
import model
import fdetection as fd
import gui

from PyQt4 import Qt, QtCore, QtGui

class Test(unittest.TestCase):


    def setUp(self):
#         self.training_set = model.TrainingSets()
#         self.face_detector = fd.FaceDetector()
#         self.gui = gui.GUI()
        app = QtGui.QApplication('')
        win = gui.GUI()
        win.show()
        app.connect(app,                             # Sender-Widget
                    Qt.SIGNAL('lastWindowClosed()'), # Signal
                    app,                             # Empfaenger
                    Qt.SLOT('quit()')                # aktivierter Slot
                    )
    def tearDown(self):
        pass


    def testImage(self):
        frame = self.gui.video.capture_next_frame()
        print type(frame)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()