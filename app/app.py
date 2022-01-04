#!/usr/bin/env python
import multiprocessing as mp
from contextlib import contextmanager

import cv2, io, traceback, time
from flask import Flask, render_template, Response
from PIL import Image
import numpy as np
from .detectors import BirdDetector, ObjectDetector

app = Flask(__name__)

VIDEO_SCREEN_SIZE = (640, 480)

# def video_processing():
#     vc = cv2.VideoCapture(0)

#     while True:
#         time.sleep(0.1)
#         _, frame = vc.read()
#         frame = cv2.resize(frame, VIDEO_SCREEN_SIZE)

#         if frame is None:
#             print("Frame is of type NoneType, reset Raspberry")

#         q1.put(frame)
#         q1_p.put(frame.copy())
#         print("Video Processing q1: ", q1.qsize())

def object_detection():
    detector = ObjectDetector()

    while True:
        print("Object detection q1 start: ", q1.qsize())
        try:
            
            frame = q1.get()
            t1_start = time.perf_counter()
            frame = detector.detect(frame)
            print("Time to detect: ", time.perf_counter()-t1_start)
            q2.put(frame)

        except Exception:
            print("----------------- 1 object detection -----------------")
            traceback.print_exc()
            print("----------------- 2 object detection -----------------")
            q2.put(np.zeros((VIDEO_SCREEN_SIZE[1], VIDEO_SCREEN_SIZE[0], 3)))
        print("Object detection q2: ", q2.qsize())

def start_api():
    app.run(host="0.0.0.0", threaded=True)


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")

@contextmanager
def get_video_capture(*args, **kwargs):
    vc = cv2.VideoCapture(0)
    try:
        yield vc
    finally: 
        vc.release()
        print("Released video capture.")

def gen():
    """Video streaming generator function."""

    vc = get_video_capture()
    frame_mask = np.zeros((VIDEO_SCREEN_SIZE[1], VIDEO_SCREEN_SIZE[0], 3))
    
    while True:
        _, frame = vc.read()
        frame = cv2.resize(frame, VIDEO_SCREEN_SIZE)

        if frame is None:
            print("Frame is of type NoneType, reset Raspberry...")

        q1.put(frame)

        if q2.qsize():
            frame_mask = q2.get()

        frame = frame + frame_mask
        frame[frame > 255] = 255

        _, image_buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(image_buffer)

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
        )

@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    q1 = mp.Queue(1)
    q2 = mp.Queue(1)

    # p1 = mp.Process(target=video_processing)
    p2 = mp.Process(target=object_detection)
    p3 = mp.Process(target=start_api)
    # p1.start()
    p2.start()
    p3.start()
