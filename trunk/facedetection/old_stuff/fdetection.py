__author__ = 'deniz'

import cv2


face = "haarcascade_frontalface_default.xml"
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
webcam = cv2.VideoCapture(0)
cv2.namedWindow("preview")
classifier = cv2.CascadeClassifier(face)

if webcam.isOpened(): # try to get the first frame
    test, frame = webcam.read()
else:
    test = False

while test:
    img = frame.copy()
    # detect faces and draw bounding boxes
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
    
    faces = classifier.detectMultiScale(g)

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


    cv2.imshow("preview", img)

    # get next frame
    test, frame = webcam.read()

    key = cv2.waitKey(20)
    if key in [27, ord('Q'), ord('q')]: # exit on ESC
        break
