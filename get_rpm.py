from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
import time 
from time import sleep
import cv2 
import matplotlib as plt
import pyfirmata
"""
This is a project that uses Sony's Camera Remote API to control an a6000. This camera 
takes photos of a DC motor over time to visually determine its RPM with respect to 
its decaying driving voltage. 
"""
"Camera Remote API by Sony"

board = pyfirmata.Arduino('COM7')

# Name pins
speed_pin = 5
direction1 = 4
direction2 = 3
voltage_read_pin = 0

# Motor Pins
# Assign mode attribute to Arduino pins 
board.digital[direction2].mode = pyfirmata.OUTPUT
board.digital[direction1].mode = pyfirmata.OUTPUT
board.digital[speed_pin].mode = pyfirmata.PWM

# Voltage reading pin
board.analog[voltage_read_pin].mode = pyfirmata.INPUT

# print(time.monotonic())
# Note that we can't use time.sleep() because no 
# ... time-tracking on the board, need to use relative time 

#! Restructure code to have a main function 
#! Make this into a function turn_on_motor(seconds = 3)

def turn_on_motor(seconds = 3, speed = 1):
    # Turn on the motor for three seconds 
    current_time = time.monotonic()
    future_time = time.monotonic() + seconds
    while future_time > current_time:
     #  print(time.monotonic())
        board.digital[direction1].write(1)
        board.digital[direction2].write(0)
        board.digital[speed_pin].write(speed)
        current_time = time.monotonic()

    # Turn off the motor 
    board.digital[direction1].write(1)
    board.digital[direction2].write(0)
    board.digital[speed_pin].write(0)


def wait(seconds = 3):
    current_time = time.monotonic()
    future_time = time.monotonic() + seconds
    while future_time > current_time:
        current_time = time.monotonic()

turn_on_motor(seconds = 5)

wait(seconds = 2)
turn_on_motor(seconds = 5, speed = .8)
board.analog[voltage_read_pin].read()
wait(seconds=2)
turn_on_motor(seconds = 5, speed = .6)
wait(seconds=2)
turn_on_motor(seconds = 5, speed = .5)


# # Code to communicate with camera 
# # camera = Camera()  # create camera instance
# camera_info = camera.info()  # get camera camera_info
# # print(camera_info)
# # print(camera.api_version)  # print api version of camera

# camera.do(Actions.actTakePicture)

# # print(Camera.info(camera))








  
