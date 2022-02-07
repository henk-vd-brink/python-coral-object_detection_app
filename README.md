## About the Project
This project contains a containerized machine learning module that runs on the Google Coral USB Accelerator https://coral.ai/products/accelerator/ in conjunction with a Raspberry Pi 4 (8GB). The video input is obtained using a USB camera. The software is capable of running any tflite model smaller than 8MB inside the USB Accelerator's memory.

Additionally, it is worth mentioning that the software runs on multiple CPU cores to optimise performance. One core is dedicated to the video stream, the other is fully dedicated to detecting.

## Getting Started
To run this instantly, make sure that the Coral USB Accelerator and USB camera are connected to the USB ports on the Raspberry Pi. For the best performance I would recommend using the USB 3.0 ports.

### Prerequisites
- Docker

### Installation
```
git clone https://github.com/henk-vd-brink/python-coral-image_detection_app.git
cd python-coral-image_detection_app
```

### Build
```
sudo docker build -t app .
```

### Run
```
sudo docker run --privileged -p 5000:5000 -v /dev/bus/usb:/dev/bus/usb app
```

The camera + detection boxes stream are exposed on port 5000.


## Live Detection Examples
### Standard pre-trained model
![recording_coco](https://user-images.githubusercontent.com/47902049/152767331-1270a1e3-5f93-44d3-a47f-848b07b36020.gif)

### Custom trained model
![recording_dice_detection](https://user-images.githubusercontent.com/47902049/152764991-9d40ceec-d52c-4a86-b1bb-81401a2efaa9.gif)


