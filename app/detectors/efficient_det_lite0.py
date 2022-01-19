from .base_detector import BaseDetector

import cv2
import numpy as np
from pycoral.utils import edgetpu
from PIL import Image


class EfficientDetLite0(BaseDetector):

    _OUTPUT_LOCATION_NAME = "location"
    _OUTPUT_CATEGORY_NAME = "category"
    _OUTPUT_SCORE_NAME = "score"
    _OUTPUT_NUMBER_NAME = "number of detections"

    _model_file = "app/detectors/assets/models/lite-model_efficientdet_lite1_dice_detection_220119.tflite"
    _label_file = "app/detectors/assets/labels/lite-model_efficientdet_lite0_dice_detection_labels.txt"

    def __init__(self):
        self._interpreter = edgetpu.make_interpreter(self._model_file)
        self._interpreter.allocate_tensors()

        with open(self._label_file, "r") as label_file:
            self._label_list = label_file.readlines()
        self._label_list = [label.replace("\n", "") for label in self._label_list]

        input_detail = self._interpreter.get_input_details()[0]
        sorted_output_indices = sorted(
            [output["index"] for output in self._interpreter.get_output_details()]
        )

        self._output_indices = {
            self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
            self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
            self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
            self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
        }

        self._input_size = input_detail["shape"][2], input_detail["shape"][1]
        self._is_quantized_input = input_detail["dtype"] == np.uint8

    def _preprocess(self, input_image):
        output_image = (
            Image.fromarray(input_image)
            .convert("RGB")
            .resize(self._input_size, Image.ANTIALIAS)
        )
        return output_image

    def _get_mask(self, boxes, classes, scores, count, image_width, image_height):

        image = np.zeros((image_height, image_width, 3))
        for i in range(count):
            if scores[i] >= 0.2:
                y_min, x_min, y_max, x_max = boxes[i]
                bb_x_min = round(x_min * image_width)
                bb_x_max = round(x_max * image_width)
                bb_y_min = round(y_min * image_height)
                bb_y_max = round(y_max * image_height)

                class_id = int(classes[i] + 1)

                image = cv2.rectangle(
                    image, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (0, 0, 255), 1
                )
                image = cv2.putText(
                    image,
                    self._label_list[class_id],
                    (bb_x_min, bb_y_min + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                image = cv2.putText(
                    image,
                    f"{round(100*scores[i])}",
                    (bb_x_max, bb_y_min + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        return image

    def detect(self, image):
        image_height, image_width, _ = image.shape

        input_tensor = self._preprocess(image)

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        return self._get_mask(boxes, classes, scores, count, image_width, image_height)

    def _set_input_tensor(self, image):
        """Sets the input tensor."""
        tensor_index = self._interpreter.get_input_details()[0]["index"]
        input_tensor = self._interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def _get_output_tensor(self, name):
        """Returns the output tensor at the given index."""
        output_index = self._output_indices[name]
        tensor = np.squeeze(self._interpreter.get_tensor(output_index))
        return tensor
