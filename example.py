#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from mtcnn.network.factory import NetworkFactory
import tensorflow as tf
from mtcnn import MTCNN

detector = MTCNN()
p_value = NetworkFactory().build_pnet()
o_value = NetworkFactory().build_onet()
r_value = NetworkFactory().build_rnet()
#
image = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)

# # Result is an array with all the bounding boxes detected.
for i in range(len(result)):
    bounding_box = result[i]['box']
    keypoints = result[i]['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
#
cv2.imwrite("appa_drawn.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# p1 = []
# p = p_value.layers[2].predict()
# o = o_value.layers
# r = r_value.layers
# model_json= r_value.to_json()
# with open("rnet.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# r_value.save_weights("rnet.h5")
# for i in range(len(p)):
#     p1.append(p[i])
# np.savetxt('p_weight.csv', , fmt='%s', delimiter=',')
# print(p)