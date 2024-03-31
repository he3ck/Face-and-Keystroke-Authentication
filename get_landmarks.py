# To install dlib, try the following commands:
    # pip install cmake
    # pip install dlib
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def distances(points):
    return [np.linalg.norm(p1 - p2) for p1 in points for p2 in points]

def get_bounding_box(rect):
    x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    return x, y, w, h

def shape_to_np(shape, num_coords, dtype="int"):
    coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(num_coords)], dtype=dtype)
    return coords

def get_landmarks(images, labels, save_directory="", num_coords=5, to_save=False):
    print("Getting %d facial landmarks" % num_coords)
    landmarks = []
    new_labels = []
    img_ct = 0
    predictor_path = 'shape_predictor_5_face_landmarks.dat' if num_coords == 5 else '../shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img, label in zip(images, labels):
        img_ct += 1
        detected_faces = detector(img, 1)
        for d in detected_faces:
            new_labels.append(label)
            x, y, w, h = get_bounding_box(d)
            points = shape_to_np(predictor(img, d), num_coords)
            dist = distances(points)
            landmarks.append(dist)
            if to_save:
                for (x_, y_) in points:
                    cv2.circle(img, (x_, y_), 1, (0, 255, 0), -1)
                plt.figure()
                plt.imshow(img)
                os.makedirs(save_directory, exist_ok=True)
                plt.savefig(os.path.join(save_directory, label + '%d.png' % img_ct))
                plt.close()
            if img_ct % 50 == 0:
                print("%d images with facial landmarks completed." % img_ct)
    return np.array(landmarks), np.array(new_labels)
