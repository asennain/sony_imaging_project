import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd 
import math


def main():

    # Create list where all images are stored
    image_list = []

    # Loading image from file path
    test_image = cv.imread(filename = r"D:\DCIM\16430414\DSC01959.JPG")

    # Resize image to (800, 1200)
    # Note that camera has 24 MP resolution 
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


    #! Test to see which threshold value yields max number of lines detected, use that threshold 
    threshold_xvals = []
    threshold_yvals = []
    for i in range(50,450,2):
        lines = cv.HoughLinesP(test_image_bin, rho = 10,theta = 1*np.pi/180,threshold = i + 1, minLineLength = 300, maxLineGap = 20)
        threshold_xvals.append(i)
        threshold_yvals.append(len(lines))
    # plt.plot(threshold_xvals, threshold_yvals)
    # plt.show()


    # List of all detected lines 
    lines_list = cv.HoughLinesP(test_image_bin, rho = 10,theta = 1*np.pi/180,threshold = 50, minLineLength = 300, maxLineGap = 20)

    # Plot all detected lines in blue
    for lines in lines_list:
        x1, y1, x2, y2 = lines[0]
        cv.line(test_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
    # Change line_lists to list of tuples, easier sorting and accessing 
    lines_list_tuples = []
    for i in range(len(lines_list)):
        x1, y1, x2, y2 = lines_list[i][0]
        lines_list_tuples.append((x1, y1, x2, y2))

    # Create dataFrame to differentiate first and second lines 
    group_lines_df = pd.DataFrame(lines_list_tuples, columns=['x1','y1', 'x2', 'y2'])

    # Create empty column used to identify lines, (using arbitrary dot product)
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

    # Appending column to DataFrame, can sort lines by angle
    group_lines_df['angle'] = angles_compare

    # Now sort the DataFrame by grouping the angles relative to comparison vector, given in angle column, reset index (shifts after .sort_values)
    group_lines_df_sorted = group_lines_df.sort_values(by = ['angle']).reset_index().drop('index', axis = 1)

    #! Now need to obtain the index where line 2 starts so we can split the DataFrame for each line
    # .idmax() returs Pandas Series object with index location of max for every column, angles max difference is 5th element in this Series 
    line2_index = int(group_lines_df_sorted.diff().idxmax()[4])
    line1_df = group_lines_df_sorted[:line2_index].reset_index().drop('index', axis = 1).drop('angle', axis = 1)
    line2_df = group_lines_df_sorted[line2_index:].reset_index().drop('index', axis = 1).drop('angle', axis = 1)

    # Highlight line #1 using average value points from respective line 1 DataFrame
    cv.line(test_image, (int(line1_df.mean(axis = 0)[0]), int(line1_df.mean(axis = 0)[1])), (int(line1_df.mean(axis = 0)[2]), int(line1_df.mean(axis = 0)[3])), (0, 255, 0), 10)
    cv.line(test_image, (int(line2_df.mean(axis = 0)[0]), int(line2_df.mean(axis = 0)[1])), (int(line2_df.mean(axis = 0)[2]), int(line2_df.mean(axis = 0)[3])), (0, 255, 0), 10)

    # Unformatted lines as list, not in np.array([xc,yc,x2,y2]) form, need center points to come first
    line1_points = line1_df.mean(axis = 0).astype(int).values.tolist()
    line2_points = line2_df.mean(axis = 0).astype(int).values.tolist()

    # Convert to np.array()
    line1_points = np.array(line1_points)
    line2_points = np.array(line2_points)

    # print(f"Unformatted points, Line 1: {line1_points} Line 2: {line2_points}\n")


    #! Important that we have points in form with center first so dot product is accurate 
    #! ... goal is to have array for each line as np.array([xc, yc, x1, y1])
    
    # Set point types based off of point_differences (identify center)
    # This is where first two points are already center points for both lines 
    center_point_first(line1_points, line2_points)

    # Make sure points reflect normal x-y plane, openCV treats top of image as zero y point, set bottom as zero 
    shift_yvals(line1_points, image_height = height)
    shift_yvals(line2_points, image_height = height)
        

    # print(f"Formatted points (center points first). Line 1: {line1_points}, Line 2: {line2_points}\n")
    # print(f"Number of lines detected: {len(lines_list)}")


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

    # Find final angle
    print(find_angle_from_lines(line1_points, line2_points))

    # Show the finished plot 
    plt.show()

    
   











def shift_yvals(line_points, image_height):
    """
    OpenCV represents y-values as top being zero, need to change
    to traditional indexing.
    """
    for i in range(len(line_points)):
        if line_points[i] not in [line_points[0], line_points[2]]:
            line_points[i] = image_height - line_points[i]
  


def find_angle_from_lines(line1, line2):
        """
        line1 and line2 are np.array([xc, yc, x2, y2])
        """
        # find difference between x and y for each line
        line1 = np.array([line1[2]-line1[0], line1[3]-line1[1]]) 
        line2 = np.array([line2[2]-line2[0], line2[3]-line2[1]]) 
        final_angle = math.acos(   np.dot(line1, line2)    /    ((np.linalg.norm(line1))*(np.linalg.norm(line2))) ) * 180/np.pi

        return round(final_angle, 2)

def center_point_first(line1_points, line2_points):
    """
    This function ensures that the array used to reprsent each line starts
    with the center point first, which is important for dot product. 
    """
    point_differences = []
    point_differences.extend(
        (
            np.sum(np.abs(line1_points[:2] - line2_points[:2])),
            np.sum(np.abs(line1_points[:2] - line2_points[2:])),
            np.sum(np.abs(line1_points[2:] - line2_points[:2])),
            np.sum(np.abs(line1_points[2:] - line2_points[2:])),
        )
    )
    # find which condition in the list sets center points based on least value in point_differences[] 
    # ... if difference between two points is very small, this is in indication those are our center points for both lines
    point_differences.index(min(point_differences))


    if point_differences.index(min(point_differences)) == 0:
    # do nothing, points are already in correct np.array([xc, yc, x1, y1]) form
        pass
    elif point_differences.index(min(point_differences)) == 1:
        # first line is in correct form, but second needs to switch first point
        line2_points = np.append(line2_points[2:], line2_points[:2])

    elif point_differences.index(min(point_differences)) == 2:
        # second line is in correct form, but first line needs to switch first point
        line1_points = np.append(line1_points[2:], line1_points[:2])

    elif point_differences.index(min(point_differences)) == 3:
        # Both lines need to be switched and redefined
        line1_points = np.append(line1_points[2:], line1_points[:2])
        line2_points = np.append(line2_points[2:], line2_points[:2])

if __name__ == "__main__":
    main()
