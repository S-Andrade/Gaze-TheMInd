sudo docker build -t gaze .
sudo docker run --rm -it --device=/dev/video0:/dev/video0  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gaze