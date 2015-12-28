import cv2, sys, numpy as np

from PyQt4 import QtCore
from PyQt4 import Qt
from PyQt4 import QtGui
from PyQt4 import QtOpenGL

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


class CameraDevice(QtCore.QObject):

    _DEFAULT_FPS = 30

    newFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraId=0, mirrored=False, parent=None):
        super(CameraDevice, self).__init__(parent)

        self.mirrored = mirrored

        self._cameraDevice = cv2.VideoCapture(cameraId)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._queryFrame)
        self._timer.setInterval(1000/self.fps)

        self.paused = False

    @QtCore.pyqtSlot()
    def _queryFrame(self):
        success, frame = self._cameraDevice.read()
        if success:
            if self.mirrored:
                frame = cv2.flip(frame, 1)
            self.newFrame.emit(frame)

    @property
    def paused(self):
        return not self._timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self._timer.recognize_face_stopped()
        else:
            self._timer.start()

    @property
    def frameSize(self):
        w = self._cameraDevice.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        h = self._cameraDevice.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return int(w), int(h)

    @property
    def fps(self):
        _fps = self._cameraDevice.get(cv2.cv.CV_CAP_PROP_FPS)
        if not _fps > 0:
            _fps = self._DEFAULT_FPS
        return _fps



class ARWidget(QtOpenGL.QGLWidget):

    newFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cameraDevice, parent=None):
        super(ARWidget, self).__init__(parent)

        self._frame = None

        self._pose = np.eye(4, dtype = np.float64)

        self._cameraDevice = cameraDevice
        self._cameraDevice.newFrame.connect(self._onNewFrame)

        #w, h = self._cameraDevice.frameSize
        w = 640
        h = 480

        if not w*h:
            w = 640
            h = 480
            raise ValueError("Incorrect image size! (An error seems to have occured with the video device)")

        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)


    def initializeGL(self):
        glViewport(0, 0, self.width(), self.height());
        glClearColor(1.0, 0.5, 0.0, 1.0)
        glClearDepth(1.0)
        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
        glEnable(GL_NORMALIZE);
        glEnable(GL_DEPTH_TEST);
        glShadeModel(GL_SMOOTH);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LEQUAL);
        glEnable(GL_LIGHT0);
        glLineWidth(3.0)


    def paintGL(self):
        if self._frame is None:
            return
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.drawFrame()
        #self.draw3DScene()

    def resizeGL(self, w, h):
        pass

    @QtCore.pyqtSlot(np.ndarray)
    def _onNewFrame(self, frame):
        self._frame = np.copy(frame)
        self.newFrame.emit(self._frame)

        ### TODO: (Ignore for assignment 3)         ###
        ### Estimate the camera/marker pose         ###
        ### For example:                            ###

        #self._pose = tracker.estimatePose(self._frame)

        #and delete this:
        self._pose[2, 3] = (self._pose[2, 3] + 1)%100

        self.updateGL()

    def draw3DScene(self):
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, float(self.width())/float(self.height()), 0.1, 1000.0)
        # Better: glMultMatrixd(tracker.getProjectionMatrix().T)
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)

        # use the etimated pose for model transformation
        glMultMatrixd(self._pose.T)

        # draw simple coordinate axes
        glBegin(GL_LINES)
        glColor3d(1.0,0.0,0.0)
        glVertex3d(0.0, 0.0, 0.0)
        glVertex3d(10.0, 0.0, 0.0)
        glColor3d(0.0,1.0,0.0)
        glVertex3d(0.0, 0.0, 0.0)
        glVertex3d(0.0, 10.0, 0.0)
        glColor3d(0.0, 0.0, 1.0)
        glVertex3d(0.0, 0.0, 0.0)
        glVertex3d(0.0, 0.0, 10.0)
        glEnd()

        # draw teapot
        glEnable(GL_LIGHTING)
        glPushMatrix()
        glTranslate(0.0, 0.0, 1.0)
        glRotate(90.0, 1.0, 0.0, 0.0)
        glutSolidTeapot(1)
        glPopMatrix()
        glDisable(GL_LIGHTING)


    def drawFrame(self):
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, self.width(), self.height(), 0.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        # convert the numpy array to an opengl texture
        glTexImage2D(GL_TEXTURE_2D, 0, 3, self._frame.shape[1], self._frame.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, self._frame.tostring());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);

        glDisable(GL_DEPTH_TEST);

        glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
        glColor3d(1.0,1.0,1.0);

        # draw the frame mapped to a textured quad
        glEnable(GL_TEXTURE_2D);
        glBegin(GL_QUADS);
        glTexCoord2f( 0.0, 0.0);
        glVertex3f( 0, 0, 0 );

        glTexCoord2f( 1.0, 0.0 );
        glVertex3f( self.width(), 0, 0 );

        glTexCoord2f( 1.0, 1.0 );
        glVertex3f( self.width(), self.height(), 0 );

        glTexCoord2f( 0.0, 1.0 );
        glVertex3f( 0, self.height(), 0 );
        glEnd();
        glDisable(GL_TEXTURE_2D);

        glEnable(GL_DEPTH_TEST);


    def changeEvent(self, e):
        if e.type() == QtCore.QEvent.EnabledChange:
            if self.isEnabled():
                self._cameraDevice.newFrame.connect(self._onNewFrame)
            else:
                self._cameraDevice.newFrame.disconnect(self._onNewFrame)


class MyMainWindow(QtGui.QWidget):
    def __init__(self):
        self.lframe = None
        QtGui.QWidget.__init__(self, None)
        self.setWindowTitle('Simple AR Display')
        #cv2.resize(self.lframe,100,100)

        # specify layout
        vbox = QtGui.QGridLayout(self)

        # get camera device
        self.cameraDevice = CameraDevice(mirrored=False)
        self.cameraDevice.newFrame.connect(self.onNewCameraFrame)

        # add widget to show the augmented video input image

        arWidget = ARWidget(self.cameraDevice)
        arWidget.newFrame.connect(self.onNewCameraFrame)
        vbox.addWidget(arWidget,0,0)


    def onNewCameraFrame(self, frame):

        # ================================ Face Recognize =========================================== #
        img =frame.copy()

        # Quadrat zum Groessenvergleich 
##        square = 400
##        cv2.rectangle(img,(320-square/2, 240-square/2),(320+square/2, 240+square/2),(255, 255, 0), 2)
##
        #face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")    # schneller angeblilch    
        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
        #mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")

        #Image fuer Gesichts Erkennung vorbereiten
        assert(img.shape[2] == 3)
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        DETECTION_WIDTH = 320
        scale = img.shape[1] / float(DETECTION_WIDTH)
        if img.shape[1] > DETECTION_WIDTH:
            scaled_height = int(img.shape[0]/scale +0.5)
            smallg = cv2.resize(g, (DETECTION_WIDTH,scaled_height))
        else:
            smallg = g
        smallg = cv2.equalizeHist(smallg)
        # Parameter erhoehen die Performance
        
        faces = face_cascade.detectMultiScale(smallg,                                                       
                                              scaleFactor = 1.2, 
                                              minNeighbors = 4, 
                                              minSize = (60, 60),
                                              maxSize = (400, 400),
                                              flags = cv2.cv.CV_HAAR_SCALE_IMAGE #| cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT | cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
                                              )

        for (x,y,w,h) in faces:
            if img.shape[1] > DETECTION_WIDTH:
                x = int(x * scale + 0.5)
                y = int(y * scale + 0.5)
                w = int(w * scale + 0.5)
                h = int(h * scale + 0.5)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = img[y:y+h, x:x+w]

            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            #mouth = mouth_cascade.detectMultiScale(g, 1.3, 5)
            
        if self.lframe is not None:
            frame[:] = self.lframe

        self.lframe = img


if __name__ == "__main__":

    app = QtGui.QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    sys.exit(app.exec_())



