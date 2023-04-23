import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd 
import math
import os

def shift_yvals(line_points, image_height):
    """
    OpenCV represents y-values as top being zero, need to change
    to traditional indexing.
    """
    for i in range(len(line_points)):
        if line_points[i] not in [line_points[0], line_points[2]]:
            line_points[i] = image_height - line_points[i]
  


def find_rpm_from_lines(line1, line2, flash_hertz):
        """
        line1 and line2 are np.array([xc, yc, x2, y2])
        """
        # find delta x and delta y for each line
        line1 = np.array([line1[2]-line1[0], line1[3]-line1[1]]) 
        line2 = np.array([line2[2]-line2[0], line2[3]-line2[1]]) 
        final_angle = math.acos(   np.dot(line1, line2)    /    ((np.linalg.norm(line1))*(np.linalg.norm(line2))) ) * 180/np.pi
        final_rpm = final_angle * flash_hertz * 60/360

        return round(final_rpm, 2)

def center_point_first(line1_points, line2_points):
    # sourcery skip: merge-list-extend, unwrap-iterable-construction
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

    return line1_points, line2_points

def get_image_data(folder, image, flash_hertz, duty_cycle):
    """
    return test_image, sorted_line1_points, sorted_line2_points, angular_velocity, duty_cycle
    """
    # Loading image from file path
    test_image = cv.imread(filename = os.path.join(folder, image))

    # Resize image to (800, 1200)
    # Note that camera has 24 MP resolution 
    scale  = 0.2
    width = int(test_image.shape[1] * scale)
    height = int(test_image.shape[0] * scale)
    dim = (width, height)
    test_image = cv.resize(test_image, dim)

    # Convert image to grayscale 
    test_image_gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)

    # Binarize image
    test_image_bin = cv.threshold(test_image_gray, 140, 255, cv.THRESH_BINARY)[1]

    # # Erode image 
    # kernel = np.ones((3,3), np.uint8)
    # test_image_bin = cv.erode(test_image_bin, kernel, iterations=3)

    # List of all detected lines 
    lines_list = cv.HoughLinesP(test_image_bin, rho = 10,theta = 1*np.pi/180,threshold = 50, minLineLength = 300, maxLineGap = 20)

    # Plot all detected lines in blue
    for lines in lines_list:
        x1, y1, x2, y2 = lines[0]
        cv.line(test_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
        
    # Change line_lists to list of tuples, easier sorting and accessing 
    lines_list_tuples = []
    for i in range(len(lines_list)):
        x1, y1, x2, y2 = lines_list[i][0]
        lines_list_tuples.append((x1, y1, x2, y2))

    # Create DataFrame to differentiate first and second lines 
    group_lines_df = pd.DataFrame(lines_list_tuples, columns=['x1','y1', 'x2', 'y2'])

    # Create empty column used to identify lines, (using arbitrary dot product)
    group_lines_df['angle'] = ''

    # To filter the data so that both lines are separated, find the dot product for 
    # ... each line in hough() to some random vector, group lines by similar angle from
    # ... taking dot product with this random vector  
    angles_compare = []
    for index, currentrow in group_lines_df.iterrows():
        x = currentrow['x2'] - currentrow['x1']
        y = currentrow['y2'] - currentrow['y1']
        # vectors for dot product 
        a = np.array([x, y])
        b = np.array([0, 1000])

        # Find comparison angle by comparing dot product against hypothetical vector (0,1000)
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

    # Highlight lines using average value points from respective line 1 DataFrame
    cv.line(test_image, (int(line1_df.mean(axis = 0)[0]), int(line1_df.mean(axis = 0)[1])), (int(line1_df.mean(axis = 0)[2]), int(line1_df.mean(axis = 0)[3])), (0, 255, 0), 1)
    cv.line(test_image, (int(line2_df.mean(axis = 0)[0]), int(line2_df.mean(axis = 0)[1])), (int(line2_df.mean(axis = 0)[2]), int(line2_df.mean(axis = 0)[3])), (0, 255, 0), 1)
    

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
    line1_points = center_point_first(line1_points, line2_points)[0]
    line2_points = center_point_first(line1_points, line2_points)[1]

    # Highlight center points for each line
    cv.circle(test_image, tuple(line1_points[:2]), radius=20, color=(255, 0, 0), thickness=-2)
    cv.circle(test_image, tuple(line2_points[:2]), radius=20, color=(255, 0, 0), thickness=-2)
    # Highlight end points for each line 
    cv.circle(test_image, tuple(line1_points[2:]), radius=20, color=(0, 255, 0), thickness=-2)
    cv.circle(test_image, tuple(line2_points[2:]), radius=20, color=(0, 255, 0), thickness=-2)

    # Make sure points reflect normal x-y plane, openCV treats top of image as zero y point, set bottom as zero 
    # shift_yvals(line1_points, image_height = height)
    # shift_yvals(line2_points, image_height = height)
        
    # print(f"Formatted points (center points first). Line 1: {line1_points}, Line 2: {line2_points}\n")
    # print(f"Number of lines detected: {len(lines_list)}")

   
    # Find final angle
    angular_velocity = find_rpm_from_lines(line1_points, line2_points, flash_hertz)
    return test_image, line1_points, line2_points, angular_velocity, duty_cycle