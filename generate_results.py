# sourcery skip: convert-to-enumerate, for-append-to-extend, list-comprehension
import matplotlib.pyplot as plt
import matplotlib
import os
from image_processing import get_image_data
from pathlib import Path
from constants import DUTY_CYCLE_LIST

#! Set folder 
folder = r"C:\Users\asenn\OneDrive\School\Courses\ME 695\Project\Test2"
images_list = os.listdir(folder)

#  Get all needed experiment data from images, assign PWM values as well to image return values 
#  return test_image, line1_points, line2_points, angular_velocity, #! adding duty_cycle to tuple in this loop
image_data =  []
i = 0
for image in images_list: 
    image_data.append(get_image_data(folder = folder, image = image, flash_hertz = 45, duty_cycle = DUTY_CYCLE_LIST[i]))
    i+=1

# image_data =  ['ya.', 'nos', 'so']

# Generate plot of images 
# fig = plt.figure(figsize=(14, 5))
num_rows = 1
num_columns = len(images_list)

# Figure for images
fig_images, ax_images = plt.subplots(num_rows, num_columns, figsize=(12,4))
# FIgure for data
fig_data, ax_data = plt.subplots(figsize=(7,5))
matplotlib.rcParams.update({'font.size': 7})

# Populate data in loop 
pwm_data = []
rpm_data = []

# Plot data
for i in range(len(image_data)):
    ax_images[i].imshow(image_data[i][0])
    ax_images[i].axis('off')
    ax_images[i].set_title(f"RPM: {int(image_data[i][3])}    PWM Duty Cycle: {int(image_data[i][4]*100)}%")
    pwm_data.append(int(image_data[i][4]*100))
    rpm_data.append(int(image_data[i][3]))

ax_data.plot(pwm_data, rpm_data, 'bo')
ax_data.set_xlabel("PWM Duty Cycle (%)")
ax_data.set_ylabel("Revolutions Per Minute (RPM)")
plt.show()