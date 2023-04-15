# sourcery skip: for-append-to-extend, list-comprehension
import matplotlib.pyplot as plt
import os
from image_processing import get_image_data
from pathlib import Path
from constants import DUTY_CYCLE_LIST

folder = r"D:\DCIM\16430414"
images_list = os.listdir(folder)

#  return test_image, line1_points, line2_points, angular_velocity
image_data =  []
for image in images_list: 
    image_data.append(get_image_data(folder = folder, image = image, flash_hertz = 45))

# image_data =  ['ya.', 'nos', 'so']

# Generate plot of images 
# fig = plt.figure(figsize=(14, 5))
num_rows = 1
num_columns = len(images_list)

fig, ax = plt.subplots(num_rows, num_columns, figsize=(16,6))

# Plot data
for i in range(len(image_data)):
    ax[i].imshow(image_data[i][0])
    ax[i].axis('off')
    ax[i].set_title(f"RPM: {image_data[i][3]}")


plt.show()