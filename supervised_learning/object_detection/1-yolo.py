#!/usr/bin/env python3
'''yolo algorithm'''
import keras as K
import numpy as np


class Yolo:
    '''yolo class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''
        -model_path is the path to where a Darknet Keras model is stored
        -classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        -class_t is a float representing the box score threshold for the
        initial filtering step
        -nms_t is a float representing the IOU threshold for non-max
        suppression
        -anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes
        '''
        self.model = K.models.load_model(model_path)
        with open(classes_path, "r") as f:
            self.class_names = f.read().split()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        '''
        -outputs is a list of numpy.ndarrays containing the predictions
        from the Darknet model for a single image
        -image_size is a numpy.ndarray containing the image's original
        size [image_height, image_width]
        '''
        img_h, img_w = image_size
        res = []
        for i, output in enumerate(outputs):
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bx = 1 / (1 + np.exp(-output[:, :, :, 0])) + img_w
            by = 1 / (1 + np.exp(-output[:, :, :, 1])) + img_h

            bw = pw * np.exp(output[:, :, :, 2])
            bh = ph * np.exp(output[:, :, :, 3])

            boxes = np.array([bx, by, bw, bh])
            box_confidence = 1 / (1 + np.exp(-output[:, :, :, 4]))
            box_class_probs = 1 / (1 + np.exp(-output[:, :, :, 5:]))
            res.append((boxes, box_confidence, box_class_probs))
        return res
