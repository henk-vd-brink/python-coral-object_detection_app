from .base_detector import BaseDetector

import os, cv2
import pathlib, numpy as np
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

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
        
        input_detail = self._interpreter.get_input_details()[0]

        self._size = common.input_size(self._interpreter)

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
        output_image = Image.fromarray(input_image).convert("RGB").resize(self._size, Image.ANTIALIAS)  
        return output_image

    def detect(self, image):
        input_tensor = self._preprocess(image) 

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()


        common.set_input(self._interpreter, image)
        classes = classify.get_classes(self._interpreter)
        
        labels = dataset.read_label_file(self._label_file)

        for c in classes:
            print(type(c))
            print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
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



