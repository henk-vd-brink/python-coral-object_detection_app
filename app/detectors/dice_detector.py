from .efficient_det_lite0 import EfficientDetLite0

import cv2, os
import numpy as np


class DiceDetector(EfficientDetLite0):

    _model_file = "app/detectors/assets/models/lite-model_efficientdet_lite0_dice_detection_500epochs.tflite"
    _label_file = "app/detectors/assets/labels/lite-model_efficientdet_lite0_dice_detection_labels.txt"

    INTERFACE_FOLDER_URI = "app/detectors/assets/images/dice_labels/"
    BLOCK_RESOLUTION = (49, 49)

    def __init__(self):
        super().__init__()
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

    def _update_sum_img_dict(self, img_dict, classes):
        classes_int = [int(class_) for class_ in classes]
        sum_classes = sum(classes_int)
        img = img_dict["empty_sum"].copy()

        font = cv2.FONT_HERSHEY_DUPLEX
        bottomLeftCornerOfText = (3, 44)
        fontScale = 1
        fontColor = (254, 254, 254)
        thickness = 2
        lineType = 1

        if sum_classes < 10:
            bottomLeftCornerOfText = (13, 40)

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
