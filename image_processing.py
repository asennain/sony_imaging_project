import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import statistics
from operator import itemgetter

# Create list where all images are stores 
image_list = []

# Loading image from file path
test_image = cv.imread(filename = r"D:\DCIM\16330413\DSC01956.JPG")

# Resize image to (800, 1200)
# Note that camera has 24 MP resolution which is too high
scale  = 0.2
width = int(test_image.shape[1] * scale)
height = int(test_image.shape[0] * scale)
dim = (width, height)
test_image = cv.resize(test_image, dim)

# Store original resized image in list 
image_list.append(test_image)

# Convert image to grayscale 
test_image_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)


# Binarize image
test_image_bin = cv.threshold(test_image_gray, 120, 255, cv.THRESH_BINARY)[1]


# Need to find edges for line detection 
edges = cv.Canny(test_image_bin, 200, 500, apertureSize=3)

# Now implement edge detection
minLineLength = 100
maxLineGap = 5

#! Plot to see which threshold value yields max number of lines detected
# threshold_xvals = []
# threshold_yvals = []
# for i in range(10,500):
#     lines = cv.HoughLinesP(test_image_bin, rho = 1,theta = 1*np.pi/180,threshold = i + 1, minLineLength = 400, maxLineGap = 1)
#     threshold_xvals.append(i)
#     threshold_yvals.append(len(lines))
# plt.plot(threshold_xvals, threshold_yvals)
# plt.show()

lines_list = cv.HoughLinesP(test_image_bin, rho = 1,theta = 1*np.pi/180,threshold = 200, minLineLength = 400, maxLineGap = 1)


# Make list of each value for analysis
x1_list_line1 = []
y1_list_line1 = []
x1_list_line2 = []
y1_list_line2 = []
x2_list_line1 = [] 
y2_list_line1 = []
x2_list_line2 = [] 
y2_list_line2 = []

#! Find which position the center point is located first 
test_position_0 = []

for i in range(len(lines_list)):
    x1, y1, x2, y2 = lines_list[i][0]
    # Will be in either position 0 or 2
    test_position_0.append(lines_list[i][0][0])
    

# if True then this position zero is x1 center and is instead position 2
if statistics.stdev(test_position_0) < 50:
    x2_position = 2
    y2_position = 3
    x1_position = 0 
    y1_position = 1
else:
    x2_position = 0
    y2_position = 1
    x1_position = 2 
    y1_position = 3

# Change line_lists to list of tuples, easier sorting and accessing 
lines_list_tuples = []
for i in range(len(lines_list)):
    x1, y1, x2, y2 = lines_list[i][0]
    lines_list_tuples.append((x1, y1, x2, y2))


# Reorder list from low to high by x2 value
lines_list_tuples = sorted(lines_list_tuples, key=itemgetter(x2_position))
print(lines_list_tuples)




# Use these values to delineate x2, y2 points of lines using comparison
# Average of first three values for first line, avg. of last three for last line
x2_comparison_line1 = (lines_list_tuples[0][x2_position]+lines_list_tuples[1][x2_position] + lines_list_tuples[2][x2_position]) / 3
y2_comparison_line1 = (lines_list_tuples[0][y2_position]+lines_list_tuples[1][y2_position] + lines_list_tuples[2][y2_position]) / 3

x2_comparison_line2 = (lines_list_tuples[-1][x2_position]+lines_list_tuples[-2][x2_position] + lines_list_tuples[-3][x2_position]) / 3
y2_comparison_line2 = (lines_list_tuples[-1][y2_position]+lines_list_tuples[-2][y2_position] + lines_list_tuples[-3][y2_position]) / 3

print(x2_comparison_line1)
print(x2_comparison_line2)

for i in range(len(lines_list)):
    x1, y1, x2, y2 = lines_list_tuples[i]

    # Comparison needs to be made to delineate second points
    # Line 1 delineation, use x2
    if abs(lines_list_tuples[i][x2_position] - x2_comparison_line1) < 100:
        x1_list_line1.append(lines_list_tuples[i][x1_position])
        y1_list_line1.append(lines_list_tuples[i][y1_position])
        x2_list_line1.append(lines_list_tuples[i][x2_position])
        y2_list_line1.append(lines_list_tuples[i][y2_position])

    elif abs(lines_list_tuples[i][x2_position] - x2_comparison_line2) < 100:
        x1_list_line2.append(lines_list_tuples[i][x1_position])
        y1_list_line2.append(lines_list_tuples[i][y1_position])
        x2_list_line2.append(lines_list_tuples[i][x2_position])
        y2_list_line2.append(lines_list_tuples[i][y2_position])


# Define function to take average of lists 
def average(list):
    return sum(list)/len(list)

# First point for both lines 
x1_line1 = average(x1_list_line1)
y1_line1 = average(y1_list_line1)
x1_line2 = average(x1_list_line2)
y1_line2 = average(y1_list_line2)
x2_line1 = average(x2_list_line1)
y2_line1 = average(y2_list_line1)
x2_line2 = average(x2_list_line2)
y2_line2 = average(y2_list_line2)

# Highlight average line #1
cv.line(test_image, (int(x1_line1), int(y1_line1)), (int(x2_line1), int(y2_line1)), (0, 0, 255), 10)

# Highlight line #2 using average values
cv.line(test_image, (int(x1_line2), int(y1_line2)), (int(x2_line2), int(y2_line2)), (0, 0, 255), 10)
print(f"Number of lines detected: {len(lines_list)}")

# Create a figure
fig = plt.figure(figsize=(6, 6), facecolor="Black")
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1,4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )


for ax, image in zip(grid, image_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(image, cmap = 'gray')

# Show the finished plot 
plt.show()
