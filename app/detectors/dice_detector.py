from .base_detector import BaseDetector

import cv2, os
import numpy as np
from pycoral.utils import edgetpu
from PIL import Image


class DiceDetector(BaseDetector):

    _OUTPUT_LOCATION_NAME = "location"
    _OUTPUT_CATEGORY_NAME = "category"
    _OUTPUT_SCORE_NAME = "score"
    _OUTPUT_NUMBER_NAME = "number of detections"

    _model_file = "app/detectors/assets/models/350_epochs_self_annotated_2.tflite"
    _label_file = "app/detectors/assets/labels/lite-model_efficientdet_lite0_dice_detection_labels.txt"

    INTERFACE_FOLDER_URI = "app/detectors/assets/images/dice_labels/"
    BLOCK_RESOLUTION = (49, 49)

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

        self._set_img_matrices_dict()

    def _set_img_matrices_dict(self):
        image_list = os.listdir(self.INTERFACE_FOLDER_URI)

        img_matrices_dict = {}
        for image_name in image_list:
            image_path = self.INTERFACE_FOLDER_URI + image_name
            image = cv2.imread(image_path)

            image_matrix = cv2.resize(image, self.BLOCK_RESOLUTION)
            dict_name = image_name.rsplit(".", 1)[0]

            img_matrices_dict[dict_name] = image_matrix
        self._img_dict = img_matrices_dict

    def _preprocess(self, input_image):
        output_image = (
            Image.fromarray(input_image)
            .convert("RGB")
            .resize(self._input_size, Image.ANTIALIAS)
        )
        return output_image

    def _draw_text(
        self,
        image,
        position,
        text,
        font=cv2.FONT_HERSHEY_PLAIN,
        font_scale=2,
        font_thickness=2,
        text_color=(1, 1, 255),
    ):
        x, y = position
        cv2.putText(
            image,
            text,
            (x, y - font_scale - 1),
            font,
            font_scale,
            text_color,
            font_thickness,
        )
        return image

    def _update_detection(self, boxes, classes, scores, count):
        updated_boxes = []
        updated_classes = []
        updated_scores = []
        updated_count = 0

        for i in range(count):
            if not (scores[i] >= 0.20):
                continue
            updated_boxes.append(boxes[i])
            updated_classes.append(classes[i])
            updated_scores.append(scores[i])
        updated_count = len(updated_boxes)

        return updated_boxes, updated_classes, updated_scores, updated_count

    def _get_mask(self, boxes, classes, scores, count, image_width, image_height):
        image = np.zeros((image_height, image_width, 3))

        for i in range(count):
            if not (scores[i] >= 0.20):
                continue

            y_min, x_min, y_max, x_max = boxes[i]
            bb_x_min = round(x_min * image_width)
            bb_x_max = round(x_max * image_width)
            bb_y_min = round(y_min * image_height)
            bb_y_max = round(y_max * image_height)

            class_id = int(classes[i] + 1)

            image = cv2.rectangle(
                image, (bb_x_min, bb_y_min), (bb_x_max, bb_y_max), (1, 1, 255), 3
            )

            class_label_position = (bb_x_min, bb_y_min)
            image = self._draw_text(
                image, class_label_position, self._label_list[class_id]
            )
        return image

    def _update_sum_img_dict(self, img_dict, classes):
        classes_int = [int(class_) for class_ in classes]
        sum_classes = sum(classes_int)
        img = img_dict["empty_sum"].copy()

        font = cv2.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = (0, 49)
        fontScale = 1
        fontColor = (254, 254, 254)
        thickness = 2
        lineType = 1

        cv2.putText(
            img,
            f"{sum_classes}",
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

        img_dict["sum"] = img
        return img_dict

    def _fill_empty_classes(self, classes, desired_length=6):
        # turn for example '['5', '3']' into '['empty','empty','empty','3','5','sum']'.
        # Desired lenght adds to the front of the list
        empty = ["empty" for _ in range(desired_length - 1 - len(classes))]
        final = empty + classes + ["sum"]
        return final

    def _get_fe_mask(self, img_resolution, classes, img_dict, n_squares=6):
        classes = [str(int(class_) + 1) for class_ in classes]

        if len(classes) > n_squares - 1:
            classes.sort(reverse=True)
            classes = classes[0 : n_squares - 1]
        classes.sort()

        img_dict = self._update_sum_img_dict(img_dict, classes)
        full_classes = self._fill_empty_classes(classes)

        mask = 255 * np.ones((img_resolution[0], img_resolution[1], 3), dtype=np.uint8)
        mask = self._update_mask(mask, full_classes, img_dict)
        mask = 255 * np.ones(mask.shape) - mask

        # red_mask = 1e-3 * np.ones(mask.shape)
        # red_mask[:, :, 0] = np.ones((mask.shape[0], mask.shape[1]))
        # mask = mask * red_mask
        return mask

    def _update_mask(self, mask, classes, img_dict, n_squares=6):
        h, w, c = mask.shape
        block_w, block_h = self.BLOCK_RESOLUTION

        distance_between_blocks = round(block_w / (n_squares * 2 + 1))
        y_coord = round(h / 100)

        for i, image in enumerate(classes):
            x_coord = round(w / 2) + (block_w + distance_between_blocks) * i
            mask[
                y_coord : y_coord + block_h, x_coord : x_coord + block_w, :
            ] = img_dict[image]
        return mask

    def detect(self, image):
        image_height, image_width, _ = image.shape

        input_tensor = self._preprocess(image)

        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

        boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
        classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
        scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
        count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

        boxes, classes, scores, count = self._update_detection(
            boxes, classes, scores, count
        )
        mask_1 = self._get_mask(
            boxes, classes, scores, count, image_width, image_height
        )

        mask_2 = self._get_fe_mask((image_height, image_width), classes, self._img_dict)
        return mask_1 + mask_2

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
