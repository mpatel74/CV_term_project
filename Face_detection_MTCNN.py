# face detection with mtcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import os
import time


F = MTCNN


def draw_image_with_boxes(filename, result_list,j):

    data = pyplot.imread(filename)

    pyplot.imshow(data)

    ax = pyplot.gca()
    # plot each box
    for result in result_list:
        x, y, width, height = result['box']

        rect = Rectangle((x, y), width, height, fill=False, color='red')

        ax.add_patch(rect)

    pyplot.savefig('Output/Face_Detection_MTCNN'+str(j)+'.png')
    pyplot.show()
    pyplot.close()

def main():

    image_list1 = [file for file in os.listdir('Dataset') if file.endswith('.jpg')]
    image_list1.sort()
    s = 'Dataset/'
    image_list = ["{}{}".format(s,i) for i in image_list1]
    j = 0
    for i in image_list:
        j = j+1
        pixels = pyplot.imread(i)

        detector = MTCNN()

        faces = detector.detect_faces(pixels)

        draw_image_with_boxes(i, faces,j)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('time taken to execute: ', end - start)