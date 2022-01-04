FROM python:3.7.12-buster

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

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update


RUN export DEBIAN_FRONTEND=noninteractive

RUN apt-get install libedgetpu1-max -yq
# RUN apt-get install python3-pycoral -y

RUN apt-get update

WORKDIR /app

ADD requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN pip3 install cmake
RUN pip3 install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

COPY . .

CMD ["python3", "-m", "app.app"]