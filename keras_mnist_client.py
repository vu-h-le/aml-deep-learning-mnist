# pylint: disable=invalid-name

import cv2
import numpy as np
import json
import requests

path_to_image = "/tmp/data/mnist_png/testing/4/6.png"
img_width, img_height = 28, 28
url = "http://127.0.0.1:32773/score"

img = cv2.imread(path_to_image)
img = cv2.resize(img, (img_width, img_height))

input_array = np.array(img).reshape((img_width,img_height,3))
input_array = np.expand_dims(input_array, axis=0)

headers = {'content-type': 'application/json'}
json_data = "{\"input_array\": " + str(input_array.tolist()) + "}"

r = requests.post(url, data=json_data, headers=headers)
r.json()
