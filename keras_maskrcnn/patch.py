import os
import math
import time
import keras
import cv2

import numpy as np
import tensorflow as tf

# import keras_retinanet
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('..', 'snapshots', 'resnet50_coco_v0.2.0.h5')
model = models.load_model(model_path, backbone_name='resnet50')

# load label to names mapping for visualization purposes
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                   7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
                   13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
                   21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
                   28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                   34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
                   39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
                   46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                   53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                   60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
                   67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator',
                   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                   79: 'toothbrush'}

# minute: the count of the videos clipped every minute
video_image_dir = 'path to the images extracted from the video'
saved_image_dir = 'path to save the processed images'
minutes = 0
batch = 10

# no person in this image
original_img = cv2.imread('foo')

def image_process(image_paths):
    '''
    read images and produce a batch
    :param image_paths: a list of image file
    :return: two 4-dim np array and the specific resize info
    '''
    images = []
    draws = []
    scales = []
    for s in image_paths:
        img = read_image_bgr(s)
        draw = img.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        img = preprocess_image(img)
        img, resize = resize_image(img)
        draws.append(draw)
        images.append(np.expand_dims(img, 0))
        scales.append(resize)

    images = np.concatenate(images, axis=0)

    return images, draws, scales


while True:


    images_dir = os.path.join(video_image_dir, '{}'.format(minutes))
    saved_dir = os.path.join(saved_image_dir, '{}'.format(minutes))

    # make sure the video for this minute has been processed
    while not os.path.exists(os.path.join(video_image_dir, '{}'.format(minutes + 1))):
        break

    image_filename = [os.path.join(images_dir, s) for s in os.listdir(images_dir) if s.endswith('.png')]
    images, draws, resize_info = image_process(image_filename)

    # boxes: the bound of the people
    boxes = []
    labels = []

    for i in range(math.ceil(len(images) / batch)):
        image_batch = images[i*batch:(i+1)*batch]
        outputs = model.predict_on_batch(image_batch)

        box, _, label, _ = outputs

        boxes.append(box)
        labels.append(label)

    boxes = np.vstack(boxes)
    labels = np.vstack(labels)

    # crop from no-person images and patched to the processed images
    for image_name, raw, scale, boxes_per_image, labels_per_image in zip(image_filename, draws, resize_info, boxes, labels):
        raw = np.copy(raw)
        boxes /= scale
        for box, label in zip(boxes_per_image, labels_per_image):
            if label == 0:
                x1, y1, x2, y2 = box
                raw[x1:x2, y1:y2] = original_img[x1:x2, y1:y2]

        cv2.imwrite(image_name.replace(video_image_dir, saved_image_dir), raw)

    minutes += 1













