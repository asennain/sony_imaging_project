import cv2 as cv
import pandas
import numpy 

# Loading image from file path 
test_image = cv.imread(filename = r"C:\Users\asenn\pycode\pyprojects\sony_imaging_project\data\test_threedots.jpg", flags= cv.IMREAD_GRAYSCALE)
print(test_image)
# print(type(test_image))
cv.imshow( "Test Image", test_image)
cv.waitKey(0)