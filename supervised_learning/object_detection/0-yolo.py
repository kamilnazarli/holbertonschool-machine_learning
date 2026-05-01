#!/usr/bin/env python3
'''yolo algorithm'''


class Yolo:
    '''yolo class'''
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        '''
        -model_path is the path to where a Darknet Keras model is stored
        -classes_path is the path to where the list of class names used
        for the Darknet model, listed in order of index, can be found
        -class_t is a float representing the box score threshold for the
        initial filtering step
        -nms_t is a float representing the IOU threshold for non-max suppression
        -anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes
        '''
        self.model = model_path
        self.class_names = classes_path
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
    