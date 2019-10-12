import cv2
import numpy as np

# Load image into seperate arrays for each channel.
re_img1 = cv2.imread('sunflower.jpg')
b, g, r = cv2.split(re_img1)



