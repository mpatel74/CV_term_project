# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

def main():

    # load the photograph
    pixels = imread('29_Students_Schoolkids_Students_Schoolkids_29_84.jpg')
    # load the pre-trained model
    classifier = CascadeClassifier('lbpcascade_frontalface.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(pixels)
    # print bounding box for each detected face
    for box in bboxes:
        # extract
        x, y, width, height = box
        x2, y2 = x + width, y + height
        # draw a rectangle over the pixels
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
    # show the image
    imshow('face detection', pixels)
    # keep the window open until we press a key
    waitKey(0)
    # close the window
    destroyAllWindows()

if __name__ == "__main__":
    main()