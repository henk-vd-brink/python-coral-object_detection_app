# run
```
sudo docker build -t ml-app .

sudo docker run --privileged -p 5000:5000 -v /dev/bus/usb:/dev/bus/usb ml-app

```