from libsonyapi.camera import Camera
from libsonyapi.actions import Actions
import time 
from time import sleep
import cv2 
import matplotlib as plt
import pyfirmata
from constants import DUTY_CYCLE_LIST

"""
This file communicates with an Arduino Uno and Sony a6000 to control a motor with PWM and take pictures 
at specific time intervals. 

This is a project that uses Sony's Camera Remote API to control an a6000. This camera 
takes photos of a DC motor over time to visually determine its RPM with respect to 
its decaying driving voltage. 
"""
"Camera Remote API by Sony"



#! Restructure code to have a main function 
#! Make this into a function turn_on_motor(seconds = 3)

def main():
    # Main experiment commands 
    # List of duty cyles to test for 
    for duty_cycle in DUTY_CYCLE_LIST:
        motor_on_snap_photo(seconds = 1, duty_cycle = duty_cycle)
        wait(seconds = 1)
        
    # End connection to board
    board.sp.close()
   


board = pyfirmata.Arduino('COM6') # Configure port
camera = Camera()  # create camera instance

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

def motor_on_snap_photo(seconds = 3, duty_cycle = 1):
    # sourcery skip: extract-duplicate-method
    """
    Controls the motor speed from 0-1 using PWM
    and takes a photo by communicating with the a6000
    """
    # Turn on the motor for three seconds 
    current_time = time.monotonic()
    future_time = time.monotonic() + seconds
    while future_time > current_time:
     #  print(time.monotonic())
        board.digital[direction1].write(1)
        board.digital[direction2].write(0)
        board.digital[speed_pin].write(duty_cycle)
        current_time = time.monotonic()

    camera.do(Actions.actTakePicture)
    # Turn off the motor 
    board.digital[direction1].write(1)
    board.digital[direction2].write(0)
    board.digital[speed_pin].write(0)



# Note that we can't use time.sleep() because no 
# ... time-tracking on the board, need to use relative time 
def wait(seconds = 3):
    """
    Tells the motor to pause using monotonic time
    """
    current_time = time.monotonic()
    future_time = time.monotonic() + seconds
    while future_time > current_time:
        current_time = time.monotonic()

if __name__ == "__main__":
    main()










  
