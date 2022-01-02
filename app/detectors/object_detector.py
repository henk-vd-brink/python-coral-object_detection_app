from .base_detector import BaseDetector

import os, cv2
import pathlib, numpy as np
from typing import List, NamedTuple
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

class Rect(NamedTuple):
    """A rectangle in 2D space."""
    left: float
    top: float
    right: float
    bottom: float

class Category(NamedTuple):
    """A result of a classification task."""
    label: str
    score: float
    index: int

class Detection(NamedTuple):
    """A detected object as the result of an ObjectDetector."""
    bounding_box: Rect
    categories: List[Category]

class ObjectDetector(BaseDetector):

    _OUTPUT_LOCATION_NAME = 'location'
    _OUTPUT_CATEGORY_NAME = 'category'
    _OUTPUT_SCORE_NAME = 'score'
    _OUTPUT_NUMBER_NAME = 'number of detections'

    _model_file = "app/detectors/assets/models/ssd_mobilenet_v1_1_metadata_1.tflite"
    _label_file = "app/detectors/assets/labels/ssd_mobilenet_v1_1_metadata_1_labels.txt"

    _mean = 127.5
    _std = 127.5

    def __init__(self):
        self._interpreter = edgetpu.make_interpreter(self._model_file)
        self._interpreter.allocate_tensors()

        with open(self._label_file, "r") as label_file:
            label_list = label_file.read().splitlines()
            self._label_list = [label.decode('ascii') for label in label_list]
        
        input_detail = self._interpreter.get_input_details()[0]
        sorted_output_indices = sorted(
            [output['index'] for output in self._interpreter.get_output_details()])
            

        self._output_indices = {
                                self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
                                self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
                                self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
                                self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
                                }

        self._input_size = input_detail['shape'][2], input_detail['shape'][1]
        self._is_quantized_input = input_detail['dtype'] == np.uint8

    def _preprocess(self, input_image):
        output_image = Image.fromarray(input_image).convert("RGB").resize(self._input_size, Image.ANTIALIAS)  
        return output_image

    def _postprocess(self, image, boxes, classes, scores, count, image_width, image_height):
        results = []
    
        for i in range(count):
            if scores[i] >= 0.1:
                y_min, x_min, y_max, x_max = boxes[i]
                bounding_box = Rect(
                        top=int(y_min * image_height),
                        left=int(x_min * image_width),
                        bottom=int(y_max * image_height),
                        right=int(x_max * image_width))
                class_id = int(classes[i])
                category = Category(
                                    score=scores[i],
                                    label=self._label_list[class_id],  # 0 is reserved for background
                                    index=class_id)
                result = Detection(bounding_box=bounding_box, categories=[category])
                results.append(result)

            sorted_results = sorted(
                                    results,
                                    key=lambda detection: detection.categories[0].score,
                                    reverse=True)
            print(sorted_results)


    def detect(self, image):

        image_height, image_width, _ = image.shape

        input_tensor = self._preprocess(image) 

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        self._postprocess(image, boxes, classes, scores, count, image_width, image_height)
        return image

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]['index']
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, name):
        """Returns the output tensor at the given index."""
        output_index = self._output_indices[name]
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor



