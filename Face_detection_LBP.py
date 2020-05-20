from cv2 import imread
from cv2 import imwrite
from cv2 import CascadeClassifier
from cv2 import rectangle
import time
import os

def main():

    image_list = [file for file in os.listdir('Dataset') if file.endswith('.jpg')]
    image_list.sort()
    # load the pre-trained model
    classifier = CascadeClassifier('lbpcascade_frontalface.xml')
    j = 0

    for i in image_list[:10]:
        j = j+1
        # load the photograph
        pixels = imread('Dataset/'+i)
        # perform face detection
        bboxes = classifier.detectMultiScale(pixels)
        # print bounding box for each detected face
        #print('Number of faces detected in '+i+' : '+str(len(bboxes)))
        for box in bboxes:
            # extract
            x, y, width, height = box
            x2, y2 = x + width, y + height
            # draw a rectangle over the pixels
            rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 2)

        imwrite('Output/Face_Detection_LBP'+str(i),pixels)



if __name__ == "__main__":

    start = time.time()
    main()
    end = time.time()
    print('time taken to execute: ',end - start)