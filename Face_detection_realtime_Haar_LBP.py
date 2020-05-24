from cv2 import CascadeClassifier
from cv2 import rectangle
import time
import cv2


def main():

    classifier1 = CascadeClassifier('haarcascade_frontalface_default.xml')
    classifier2 = CascadeClassifier('lbpcascade_frontalface.xml')


    video_capture = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        start = time.time()
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces1,p,confidence1 = classifier1.detectMultiScale3(gray,scaleFactor=1.05,minNeighbors= 8,outputRejectLevels = True)

        # Draw a Green rectangle around the faces identified by HAAR_Cascade
        for (x, y, w, h),conf1 in zip(faces1,confidence1):
            if conf1>1:
                rectangle(frame, (x, y), (x+w, y+h), (0,255,0),2)



        faces2,q,confidence2 = classifier2.detectMultiScale3(gray,scaleFactor=1.05,minNeighbors= 8,outputRejectLevels = True)

        # Draw a Blue rectangle around the faces identified by LBP
        for (x, y, w, h),conf2 in zip(faces2,confidence2):
            if conf2>1:
                rectangle(frame, (x, y), (x + w, y + h),(255,0,0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end = time.time()
        print('time taken to execute one frame ', end - start)

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":


    main()

