#%% Color selection
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from numpy.polynomial.polynomial import Polynomial as poly

image = mpimg.imread('test.jpg')
#print(f"Type: {type(image)}, and dimensions: {image.shape}")
image_copy = image.copy()
#print(f"Type: {type(image_copy)}, and dimensions: {image_copy.shape}")
thresh_idx = (image[:,:,0] < 240) | (image[:,:,1] < 240) | (image[:,:,2] < 240)
image_copy[thresh_idx] = [0,0,0]
plt.imshow(image_copy)
mpimg.imsave("image_color_selection.jpg",image_copy)


# %% Region mask
image_region = image_copy.copy()
xsize = image_region.shape[0]
ysize = image_region.shape[1]
left_buttom = [xsize, 0]
right_buttom = [xsize, ysize]
apex = [3*xsize/5, ysize/2]
# Fit lines using Polynomial.fit 
left_fit = poly.fit((left_buttom[0], apex[0]), (left_buttom[1], apex[1]), 1, [])
right_fit = poly.fit((right_buttom[0], apex[0]), (right_buttom[1], apex[1]), 1, [])
#buttom_fit = poly.fit((left_buttom[0], right_buttom[0]), (left_buttom[1], right_buttom[1]), 0, [])
X, Y = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize), sparse = False, indexing = 'ij') # xy indexing by default means you should make it ij to get X = len(x)*len(y) otherwise you get X = len(y)*len(x)
region_mask = (Y > left_fit(X)) & (Y < right_fit(X)) & (X < left_buttom[0])
image_region[~region_mask] = [50, 50, 250]
plt.imshow(image_region)
mpimg.imsave("image_region_mask.jpg",image_region)


# %%
