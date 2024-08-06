import numpy as np

from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

import cv2

from script.fx import prewhiten, l2_normalize

from keras.models import load_model

from scipy.spatial import distance

from mtcnn.mtcnn import MTCNN



checker_dict = {}

appearance_counter = 0



model_path = './data/model/facenet_keras.h5'

face_cascade_path = './data/cascade/haarcascade_frontalface_default.xml'

font_path = './data/font/Calibri Regular.ttf'

embedding_path = './data/arrays/embeddings.npz'

vars_path = './data/arrays/vars.npz'

detector = MTCNN()

model = load_model(model_path)

face_cascade = cv2.CascadeClassifier(face_cascade_path)

loaded_embeddings = np.load(embedding_path)

embeddings, names = loaded_embeddings['a'], loaded_embeddings['b']

loaded_vars = np.load(vars_path)

slope, intercept = loaded_vars['a'], loaded_vars['b']



camera = cv2.VideoCapture("output.avi")

fps = int(camera.get(cv2.CAP_PROP_FPS))

w = int(camera.get(3))

h = int(camera.get(4))

fourcc = cv2.VideoWriter_fourcc(*'MJPG')

out = cv2.VideoWriter('output_final.avi', fourcc, fps, (w, h))

while 1:

    var, frame = camera.read()

    if not var:

        break

    result = detector.detect_faces(frame)

    faces = []

    for target in result:

        if target['confidence'] > 0.8:

            bounding_box = target['box']

            faces.append([int(bounding_box[0]), int(bounding_box[1]),

                          int(abs(bounding_box[2])), int(abs(bounding_box[3]))])

    for (x_face, y_face, w_face, h_face) in faces:



        # Margins for Face box

        dw = 0.1*w_face

        dh = 0.2*h_face



        dist = []

        for i in range(len(embeddings)):

            try:

                dist.append(distance.euclidean(l2_normalize(model.predict(prewhiten(cv2.resize(

                    frame[y_face:y_face+h_face, x_face:x_face+w_face], (160, 160)).reshape(-1, 160, 160, 3)))), embeddings[i].reshape(1, 128)))

            except:

                pass

        try:

            dist = np.array(dist)

            if dist.min() > 1:

                name = '0'

            else:

                name = names[dist.argmin()]

                name = name.split(".")[0]

            print(name, dist.min())

            if checker_dict.get(str(name)) == None:

                appearance_counter = appearance_counter+1

                print(name)

                checker_dict[str(name)] = "True"



            if name != '0':

                font_size = int(slope[dist.argmin()] *

                                ((w_face+2*dw)//3)*2+intercept[dist.argmin()])

            else:

                font_size = int(0.1974311*((w_face+2*dw)//3)

                                * 2+0.03397702412218706)



            font = ImageFont.truetype(font_path, font_size)

            size = font.getsize(name)

            cv2.rectangle(frame, (int(x_face), int(y_face)),

                          (int(x_face+w_face), int(y_face+h_face)), (255, 255, 0), 1)

            cv2.putText(frame, str(name), (int(x_face+w_face-20), int(y_face-5)),

                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        except:

            pass

    cv2.putText(frame, "Count: "+str(appearance_counter), (int(w-200), int(50)),

                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

    #cv2.imshow('Frame', cv2.resize(frame, (w, h)))

    out.write(frame)

    if cv2.waitKey(1) & 255 == ord('q'):

        break



camera.release()

cv2.destroyAllWindows()


