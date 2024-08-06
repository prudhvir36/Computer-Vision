import os
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

images_path = '../data/images/'


def detect_faces():
    os.chdir(images_path)

    if len(os.listdir()) == 0:
        print('\n\nNo Images Found')
        input()
        quit()
    for i in os.listdir():
        name = i
        img = cv2.imread(i)
        result = detector.detect_faces(img)
        faces = []
        for target in result:
            if target['confidence'] > 0.5:
                bounding_box = target['box']
                faces.append([int(bounding_box[0]), int(bounding_box[1]),
                              int(abs(bounding_box[2])), int(abs(bounding_box[3]))])

        for (x, y, w, h) in faces:
            image = img[y:y+h, x:x+w]
            output_name = "{}.faces{}x{}_{}x{}.jpg".format(name, x, y, w, h)
            cv2.imwrite('../faces/'+output_name, image)
            print('Done Detecting: ', output_name)
    os.chdir('../../script')
