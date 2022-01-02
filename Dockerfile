FROM python:3.9.2-buster

RUN apt-get update

RUN apt-get install ffmpeg libsm6 libxext6 python3-opencv \
    build-essential cmake pkg-config \
    libjpeg-dev libtiff5-dev libpng-dev \ 
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libfontconfig1-dev libcairo2-dev \
    libgdk-pixbuf2.0-dev libpango1.0-dev \
    libgtk2.0-dev libgtk-3-dev \
    libatlas-base-dev gfortran \
    libhdf5-dev libhdf5-serial-dev libhdf5-103 \
    libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 -y

RUN apt-get install python3-pycoral

WORKDIR /app

ADD requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install cmake

COPY . .

CMD ["python3", "-m", "app"]