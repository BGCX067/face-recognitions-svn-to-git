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

        w, h = self._cameraDevice.frameSize

        if not w*h:
            w = 640
            h = 480
            raise ValueError("Incorrect image size! (An error seems to have occured with the video device)")

        self.setMinimumSize(w, h)
        self.setMaximumSize(w, h)

    #def __init__(self, image, parent=None):
    #    super(ARWidget, self).__init__(parent)

    #    self._frame = image
    #    w = 640
    #    h = 480
    #    self.setMinimumSize(w, h)
    #    self.setMaximumSize(w, h)

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
        self.gray = False
        self.invert = False
        self.grws = False
        self.shw = False
        self.shwo = False
        self.hista = False
        #self.video=True
        #self.image = False
        self.brightnessContrast = False
        self.erbsenzaehler = False

        QtGui.QWidget.__init__(self, None)
        self.setWindowTitle('Simple AR Display')
        #self.grafik = QtGui.QIMage("confetti.jpg")
        # specify layout
        vbox = QtGui.QGridLayout(self)

        # get camera device
        self.cameraDevice = CameraDevice(mirrored=False)
        self.cameraDevice.newFrame.connect(self.onNewCameraFrame)

        # add widget to show the augmented video input image

        arWidget = ARWidget(self.cameraDevice)
        arWidget.newFrame.connect(self.onNewCameraFrame)
        vbox.addWidget(arWidget,0,0)

        #image=cv2.imread('smarties.jpg')
        #arWidget = ARWidget(image)
        #vbox.addWidget(arWidget,0,0)

        # add slider to control vertical position
        self.w, self.h = self.cameraDevice.frameSize
        self.verticalPositionSlider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.verticalPositionSlider.setRange(1,self.h)
        self.verticalPositionSlider.setSingleStep(2);
        self.verticalPositionSlider.valueChanged[int].connect(self.verticalPositionSlidervalue)
        self.verticalPositionSlider.setValue(self.h/2)
        vbox.addWidget(self.verticalPositionSlider,0,1)

        # add slider to control horizontal position
        self.w, self.h = self.cameraDevice.frameSize
        self.horizontalPositionSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.horizontalPositionSlider.setRange(1,self.w)
        self.horizontalPositionSlider.setSingleStep(2);
        self.horizontalPositionSlider.valueChanged[int].connect(self.horizontalPositionSlidervalue)
        self.horizontalPositionSlider.setValue(self.w/2)
        vbox.addWidget(self.horizontalPositionSlider)

        # add slider to control horizontal position
        self.w, self.h = self.cameraDevice.frameSize
        self.verticalSizeSlider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.verticalSizeSlider.setRange(1,self.h)
        self.verticalSizeSlider.setSingleStep(2);
        self.verticalSizeSlider.valueChanged[int].connect(self.verticalSizeSlidervalue)
        self.verticalSizeSlider.setValue(self.h/2)
        vbox.addWidget(self.verticalSizeSlider,0,2)

        #add slider to control horizontal position
        self.w, self.h = self.cameraDevice.frameSize
        self.verticalSchwellwertSlider = QtGui.QSlider(QtCore.Qt.Vertical)
        self.verticalSchwellwertSlider.setRange(0,255)
        self.verticalSchwellwertSlider.setSingleStep(2);
        self.verticalSchwellwertSlider.valueChanged[int].connect(self.verticalSchwellwertSlidervalue)
        self.verticalSchwellwertSlider.setValue(127)
        vbox.addWidget(self.verticalSchwellwertSlider,0,3)

        # add Button Grey
        self.button = QtGui.QPushButton('Grey',self)
        self.button.clicked.connect(self.handleclick)
        vbox.addWidget(self.button)

        # add Button invert
        self.invertButton = QtGui.QPushButton('Invert')
        self.invertButton.clicked.connect(self.handleroibutton)
        vbox.addWidget(self.invertButton)

        # add Button Grauwertspreizung
        self.grwsButton = QtGui.QPushButton('Gauwertspreizung')
        self.grwsButton.clicked.connect(self.handleGrwButton)
        vbox.addWidget(self.grwsButton)

        # add Button Schwellwert button
        self.shwButton = QtGui.QPushButton('Schwellwert')
        self.shwButton.clicked.connect(self.handleSchwellwertButton)
        vbox.addWidget(self.shwButton)

        # add Button Schwellwert nach Otsu button
        self.shwoButton = QtGui.QPushButton('Schwellwert Otsu')
        self.shwoButton.clicked.connect(self.handleSchwellwertOtsuButton)
        vbox.addWidget(self.shwoButton)


        # add Histogramm Ausgleich button
        self.histaButton = QtGui.QPushButton('histogrammausgleich')
        self.histaButton.clicked.connect(self.handleHistogrammAusgleichButton)
        vbox.addWidget(self.histaButton)

        # add Brightness and contrast
        self.brightnessContrastButton = QtGui.QPushButton('BrightnessContrast')
        self.brightnessContrastButton.clicked.connect(self.handleBrightnessContrastButton)
        vbox.addWidget(self.brightnessContrastButton)

        # add Video Button
        self.videoButton = QtGui.QPushButton('Video')
        self.videoButton.clicked.connect(self.handlervideoButton)
        vbox.addWidget(self.videoButton)

        # add Image Button
        self.imageButton = QtGui.QPushButton('Image')
        self.imageButton.clicked.connect(self.handlerimageButton)
        vbox.addWidget(self.imageButton)

        # add Erbsenzaehler
        self.erbsenzaehlerButton = QtGui.QPushButton('Erbsenzaehler')
        self.erbsenzaehlerButton.clicked.connect(self.handlerbsenzaehlerButton)
        vbox.addWidget(self.erbsenzaehlerButton)


    def verticalPositionSlidervalue(self, value):
        self.verticalPositionvalue = value


    def horizontalPositionSlidervalue(self,value):
        self.horizontalPositionvalue=value

    def verticalSizeSlidervalue(self,value):
        self.verticalSizevalue=value

    def verticalSchwellwertSlidervalue(self,value):
        self.verticalSchwellwertvalue = value

    def handleclick(self):
        if self.gray == True:
            self.gray = False
        else:
            self.gray = True

    def handleroibutton(self):
        if self.invert == True:
            self.invert = False
        else:
            self.invert = True

    def handleGrwButton(self):

        if self.grws == True:
            self.grws = False
        else:
            self.grws = True

    def handleSchwellwertButton(self):

        if self.shw == True:
            self.shw = False
        else:
            self.shw = True

    def handleHistogrammAusgleichButton(self):

        if self.hista == True:
            self.hista = False
        else:
            self.hista= True


    def handleSchwellwertOtsuButton(self):

        if self.shwo == True:
            self.shwo = False
        else:
            self.shwo = True

    def handleBrightnessContrastButton(self):

        if self.brightnessContrast== True:
            self.brightnessContrast = False
        else:
            self.brightnessContrast = True

    def handlerbsenzaehlerButton(self):
        if self.erbsenzaehler == True:
            erbsenzaehler = False
        else:
            erbsenzaehler = True

    def handlervideoButton(self):
        if self.video == True:
            self.video = False
        else:
            self.video = True

    def handlerimageButton(self):
        if self.image == True:
            self.image = False
        else:
            self.image = True

    def histogramm(self,frame):
        h = np.zeros((300,256,3))
        im = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        for x,y in enumerate(hist):
            cv2.line(h,(x,0),(x,y),(255,255,255))

        for x,y in enumerate(cdf_normalized):
            cv2.line(h,(x,y),(x,y),(255,0,0))
        y = np.flipud(h)
        return y


    def onNewCameraFrame(self, frame):
        # set vertical position of rectangle
        #verticalPosition = int(self.verticalPositionvalue)
        #horizontalPosition = int(self.horizontalPositionvalue)
        #verticalSize = int(self.verticalSizevalue)
        #verticalSchwellwert = int(self.verticalSchwellwertvalue)

        alpha = 2.2
        beta = 50
        """
        # Das funktioniert
        roi =frame[self.h-verticalPosition : self.h-verticalPosition+verticalSize,self.w-horizontalPosition:self.w-horizontalPosition+verticalSize]
        if self.invert:
            roi=cv2.bitwise_not(roi)

        cv2.rectangle(frame,(self.w-horizontalPosition, self.h-verticalPosition),(self.w-horizontalPosition+verticalSize, self.h-verticalPosition+verticalSize),(255, 255, 255))
        frame[self.h-verticalPosition : self.h-verticalPosition+verticalSize,self.w-horizontalPosition:self.w-horizontalPosition+verticalSize] = roi
        """
        aframe =frame.copy()

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        img = aframe
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(g, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = g[y:y+h, x:x+w]

            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        aframe= img


        """
        # In Grau umwandeln
        if self.gray:
            aframe = cv2.cvtColor(aframe,cv2.COLOR_BGR2GRAY)
            #aframe = cv2.cvtColor(aframe,cv2.COLOR_GRAY2BGR)
            #Grauwertspreizung
            if self.grws:
                aframemin = float(aframe[:].min())
                aframemax = float(aframe[:].max())
                aframe[:]= (aframe[:]-aframemin)*(255.0/(aframemax-aframemin))
            if self.shw:
                ret, aframe = cv2.threshold(aframe,verticalSchwellwert,255,cv2.THRESH_BINARY)

            if self.shwo:
                ret2,aframe = cv2.threshold(aframe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            if self.hista:
                #equ =cv2.equalizeHist(aframe)
                #aframe =np.hstack((aframe,equ))
                aframe = cv2.equalizeHist(aframe)


            hist=self.histogramm(self.lframe)
            cv2.imshow('hist',hist)
            aframe = cv2.cvtColor(aframe,cv2.COLOR_GRAY2BGR)

        #if self.brightnessContrast:
        #    aframe = alpha*aframe+beta
        """


        if self.lframe is not None:
            frame[:] = self.lframe

        self.lframe = aframe



        ### TODO:                                    ###
        ### 1. set horizontal position of box        ###
        ### 2. set size of box                       ###
        ### 3. Inverting image region in ROI defined ###
        ###    by the box                            ###


if __name__ == "__main__":

    glutInit(sys.argv) # don't need this under Mac OS X
    app = QtGui.QApplication(sys.argv)
    w = MyMainWindow()
    w.show()
    sys.exit(app.exec_())



