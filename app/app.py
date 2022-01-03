#!/usr/bin/env python
import multiprocessing as mp

import cv2, io, traceback, time
from flask import Flask, render_template, Response
from PIL import Image

from .detectors import BirdDetector, ObjectDetector

app = Flask(__name__)

vc = cv2.VideoCapture(0)

def video_processing():

    while True:
        time.sleep(1)
        _, frame = vc.read()
        q1.put(frame)
        print("joe")
        print("Video Processing q1: ", q1.qsize())

def object_detection():
    detector = ObjectDetector()

    while True:
        print("Object detection q1 start: ", q1.qsize())
        try:
            frame = q1.get()
            frame = detector.detect(frame)
            q2.put(frame)
        except Exception:
            print("Could not get frame")

        print("Object detection q1: ", q2.qsize())

def start_api():
    app.run(host="0.0.0.0", threaded=True)


@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


def gen():
    """Video streaming generator function."""

    while True:
        print("GEN!")
        frame = q2.get()

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
    q1 = mp.Queue()
    q2 = mp.Queue()

    p1 = mp.Process(target=video_processing)
    p2 = mp.Process(target=object_detection)
    p3 = mp.Process(target=start_api)
    p1.start()
    p2.start()
    p3.start()
