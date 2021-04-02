#! /usr/bin/env python3
#%% Color selection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly
import cv2 as cv
#%%
image = mpimg.imread('test.jpg')
#print(f"Type: {type(image)}, and dimensions: {image.shape}")
image_copy = image.copy()
#print(f"Type: {type(image_copy)}, and dimensions: {image_copy.shape}")
thresh_idx = (image[:,:,0] < 200) | (image[:,:,1] < 200) | (image[:,:,2] < 200)
image_copy[thresh_idx] = [0,0,0]
plt.imshow(image_copy)
#mpimg.imsave("image_color_selection.jpg",image_copy)


#%% Region mask
image_region = image_copy.copy()
xsize = image_region.shape[0]
ysize = image_region.shape[1]
left_buttom = [xsize, 0]
right_buttom = [xsize, ysize]
apex = [4.2*xsize/7, ysize/2]
# Fit lines using Polynomial.fit 
left_fit = poly.fit((left_buttom[0], apex[0]), (left_buttom[1], apex[1]), 1, [])
right_fit = poly.fit((right_buttom[0], apex[0]), (right_buttom[1], apex[1]), 1, [])
#buttom_fit = poly.fit((left_buttom[0], right_buttom[0]), (left_buttom[1], right_buttom[1]), 0, [])
X, Y = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize), sparse = False, indexing = 'ij') # xy indexing by default means you should make it ij to get X = len(x)*len(y) otherwise you get X = len(y)*len(x)
region_mask = (Y > left_fit(X)) & (Y < right_fit(X)) & (X < left_buttom[0])
image_region[~region_mask] = [50, 50, 250]
plt.imshow(image_region)
#mpimg.imsave("image_region_mask.jpg",image_region)

#%%
image_lines = image.copy()
image_lines[~thresh_idx & region_mask] = [255, 0, 0]
plt.imshow(image_lines)
#%% Canny edge detection
image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
def Canny_thresh(thresh):
    edges = cv.Canny(image_gray, thresh, 3*thresh)
    cv.imshow('Canny_edges', edges)
cv.namedWindow('Canny_edges')
cv.createTrackbar('low_thresh','Canny_edges', 140, 200, Canny_thresh) # 140 is good
Canny_thresh(0)
while(True):
    thresh = cv.getTrackbarPos('low_thresh', 'Canny_edges')
    k = cv.waitKey(1) & 0xFF
    if k != 0xFF:
        break

#%% Hough lines detection
edges = cv.Canny(image_gray, thresh, 3*thresh)
rho_res = 1
theta_res = np.pi/180
hough_inter_min = 3  #min Hough lines intersection to consider 
min_line_len = 3 #low thresh of line length
max_line_gap = 6 #max dist between 2 points that intersect in hough to get connected in one line

def Hough_thresh(call_input):
    lines = cv.HoughLinesP(edges, rho_res, theta_res, hough_inter_min, np.array([]), min_line_len, max_line_gap)
    line_image = image.copy()*0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv.line(line_image, (x1,y1), (x2,y2), [0,0,255], 5, cv.LINE_AA)
    edges_3 = np.dstack((edges, edges, edges)) 
    edges_lines = cv.addWeighted(edges_3, 0.5, line_image, 0.5, 0)
    cv.imshow('Hough_lines', edges_lines)

cv.namedWindow('Hough_lines')
cv.createTrackbar('rho_res','Hough_lines', 1, 10, Hough_thresh) 
cv.createTrackbar('theta_res*(np.pi/180)','Hough_lines', 1, 10, Hough_thresh)
cv.createTrackbar('hough_inter_min','Hough_lines', 3, 5, Hough_thresh)
cv.createTrackbar('min_line_len','Hough_lines', 3, 20, Hough_thresh)
cv.createTrackbar('max_line_gap','Hough_lines', 6, 10, Hough_thresh)
Hough_thresh(0)
while(True):
    rho_res         = cv.getTrackbarPos('rho_res', 'Hough_lines')                           #1
    theta_res       = (np.pi/180)*cv.getTrackbarPos('theta_res*(np.pi/180)', 'Hough_lines') #1
    hough_inter_min = cv.getTrackbarPos('hough_inter_min', 'Hough_lines')                   #3
    min_line_len    = cv.getTrackbarPos('min_line_len', 'Hough_lines')                      #3
    max_line_gap    = cv.getTrackbarPos('max_line_gap', 'Hough_lines')                      #6
    k = cv.waitKey(1) & 0xFF
    if k != 0xFF:
        break

