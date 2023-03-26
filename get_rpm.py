from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
from time import sleep
import cv2 
import matplotlib as plt
"""
This is a project that uses Sony's Camera Remote API to control an a6000. This camera 
takes photos of a DC motor over time to visually determine its RPM with respect to 
its decaying driving voltage. 
"""
"Camera Remote API by Sony"


# Code to communicate with camera 
camera = Camera()  # create camera instance
camera_info = camera.info()  # get camera camera_info
# print(camera_info)
# print(camera.api_version)  # print api version of camera

camera.do(Actions.actTakePicture)

# print(Camera.info(camera))








  
