#!/usr/bin/env python
import cv2, io, traceback, time
from flask import Flask, render_template, Response
from PIL import Image

from .detectors import BirdDetector, ObjectDetector

app = Flask(__name__)

detector = ObjectDetector()

vc = cv2.VideoCapture(0)

@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


def gen():
    """Video streaming generator function."""

    while True:
        _, frame = vc.read()
        # frame = cv2.resize(frame, (320, 320))
        # print("in ", frame.shape)
        # try:
        #     frame = detector.detect(frame)
        # except Exception:
        #     print(traceback.format_exc())
        # print("out ", frame.shape)

        _, image_buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(image_buffer)

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
