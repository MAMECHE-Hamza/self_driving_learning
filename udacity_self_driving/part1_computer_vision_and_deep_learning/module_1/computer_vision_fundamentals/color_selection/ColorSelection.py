#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image = mpimg.imread('test.jpg')
#print(f"Type: {type(image)}, and dimensions: {image.shape}")
image_copy = image.copy()
#print(f"Type: {type(image_copy)}, and dimensions: {image_copy.shape}")
thresh_idx = (image[:,:,0] < 240) | (image[:,:,1] < 240) | (image[:,:,2] < 240)
image_copy[thresh_idx] = [0,0,0]
plt.imshow(image_copy)
mpimg.imsave("image_color_selection.jpg",image_copy)


# %%
