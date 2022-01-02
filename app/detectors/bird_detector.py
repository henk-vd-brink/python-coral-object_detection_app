from .base_detector import BaseDetector

import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

class BirdDetector(BaseDetector):

    _model_file = "app/detectors/assets/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
    _label_file = "app/detectors/assets/labels/inat_bird_labels.txt"

    def __init__(self):
        self._interpreter = edgetpu.make_interpreter(self._model_file)
        self._interpreter.allocate_tensors()

        self._size = common.input_size(self._interpreter)

    def detect(self, image):
        pil_image = Image.fromarray(image).convert("RGB").resize(self._size, Image.ANTIALIAS)   
        common.set_input(self._interpreter, pil_image)
        self._interpreter.invoke()
        classes = classify.get_classes(self._interpreter, top_k=1)
        
        labels = dataset.read_label_file(self._label_file)

        for c in classes:
            print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
        return image



