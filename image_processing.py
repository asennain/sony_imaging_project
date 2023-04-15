import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd 
import math

# Create list where all images are stores 
image_list = []

# Loading image from file path
test_image = cv.imread(filename = r"D:\DCIM\16430414\DSC01960.JPG")

# Resize image to (800, 1200)
# Note that camera has 24 MP resolution which is high
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

#! Plot to see which threshold value yields max number of lines detected
# Automatically choose most optimal threshold value for cv.HoughLinesP()
# ... Longer processing but better quality lines
threshold_xvals = []
threshold_yvals = []
for i in range(10,400, 10):
    lines = cv.HoughLinesP(test_image_bin, rho = 1,theta = 1*np.pi/180,threshold = i + 1, minLineLength = 400, maxLineGap = 1)
    threshold_xvals.append(i)
    threshold_yvals.append(len(lines))
optimal_threshold = max(threshold_xvals)
# plt.plot(threshold_xvals, threshold_yvals)
# plt.show()

# Plot detected lines onto original image
lines_list = cv.HoughLinesP(test_image_bin, rho = 5,theta = 1*np.pi/180,threshold = optimal_threshold, minLineLength = 200, maxLineGap = 1)
for lines in lines_list:
    x1, y1, x2, y2 = lines[0]
    # Plot all detected lines in red
    cv.line(test_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    
# Change line_lists to list of tuples, easier sorting and accessing 
lines_list_tuples = []
for i in range(len(lines_list)):
    x1, y1, x2, y2 = lines_list[i][0]
    lines_list_tuples.append((x1, y1, x2, y2))

# Create dataFrame to separate lines 
group_lines_df = pd.DataFrame(lines_list_tuples, columns=['x1','y1', 'x2', 'y2'])

# Create empty column to store angles 
group_lines_df['angle'] = ''


angles_compare = []
for index, currentrow in group_lines_df.iterrows():
    x = currentrow['x2'] - currentrow['x1']
    y = currentrow['y2'] - currentrow['y1']
    # vectors for dot product 
    a = np.array([x, y])
    b = np.array([100, 100])

    # Find comparison angle by comparing dot product against hypothetical vector (100,100)
    angles_compare.append(math.acos(   np.dot(a, b)    /    ((np.linalg.norm(a))*(np.linalg.norm(b)))     ))

group_lines_df['angle'] = angles_compare
    

# Now sort the DataFrame by grouping the angles relative to comparison vector, given in angle column, reset index because it shifts 
group_lines_df_sorted = group_lines_df.sort_values(by = ['angle']).reset_index().drop('index', axis = 1)




#! Now need to obtain the index where line 2 starts 
# .idmax() returs Pandas Series object with index location of max for every column, angles max difference is 5th element in this Series 
line2_index = int(group_lines_df_sorted.diff().idxmax()[4])
line1_df = group_lines_df_sorted[:line2_index].reset_index().drop('index', axis = 1).drop('angle', axis = 1)
line2_df = group_lines_df_sorted[line2_index:].reset_index().drop('index', axis = 1).drop('angle', axis = 1)



# Highlight line #1 using average value points
# cv.line(test_image, int(line2_df.mean(axis = 0)[0]), int(line2_df.mean(axis = 0)[1]), int(line2_df.mean(axis = 0)[2]), int(line2_df.mean(axis = 0)[3]), (0, 0, 255), 10)
cv.line(test_image, (int(line2_df.mean(axis = 0)[0]), int(line2_df.mean(axis = 0)[1])), (int(line2_df.mean(axis = 0)[2]), int(line2_df.mean(axis = 0)[3])), (0, 255, 0), 10)
cv.line(test_image, (int(line1_df.mean(axis = 0)[0]), int(line1_df.mean(axis = 0)[1])), (int(line1_df.mean(axis = 0)[2]), int(line1_df.mean(axis = 0)[3])), (0, 255, 0), 10)


print(f"Number of lines detected: {len(lines_list)}")

# # Create a figure
fig = plt.figure(figsize=(6, 6), facecolor="Black")
grid = ImageGrid(
                    fig, 
                    111,  # similar to subplot(111)
                    nrows_ncols=(1,4),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.

                 )

# Plot them bad boys (processed images)
for ax, image in zip(grid, image_list):
    # Iterating over the grid returns the Axes.
    ax.imshow(image, cmap = 'gray')

# Show the finished plot 
plt.show()
