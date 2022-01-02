#!/usr/bin/env python
import cv2, io
from flask import Flask, render_template, Response

from .detectors import ObjectDetector

app = Flask(__name__)
detector = ObjectDetector()

@app.route("/")
def index():
    """Video streaming home page."""
    return render_template("index.html")


def gen():
    """Video streaming generator function."""

    vc = cv2.VideoCapture(0)

    while True:
        _, frame = vc.read()
        frame = detector.detect(frame)

        _, image_buffer = cv2.imencode(".jpg", frame)
        io_buf = io.BytesIO(image_buffer)

        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + io_buf.read() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
