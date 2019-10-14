import cv2
import numpy as np
import matplotlib.pyplot as plt

def pca(array):

    # Find mean in dataset.
    mean = 0
    for i in range(0, len(array)):
        mean += sum(array[i]) / len(array[i])
    
    mean /= len(array)
    print("\tmean: ", mean)

    # Find mean in dataset.
    mean = 0
    for i in range(0, len(array)):
        mean += sum(array[i]) / len(array[i])
    
    mean /= len(array)
    print("\tmean: ", mean)

    # Subtract mean from dataset.
    for i in range(0, len(array)):
        for j in range(0, len(array[i])):
            array[i][j] -= mean

    # Calculate covarience matrix.
    

    # Calculate unit eigenvectors and eigen values of covariance matrix.

    # Order eigen vectors by its eigen value, highest to low.




# Load image into seperate arrays for each channel.
inImage = cv2.imread('sunflower.jpg')
b, g, r = cv2.split(inImage)
# Make sure we have ints, not utins, since subtracting may cause negative values.
b = np.asarray(b, dtype=int)
g = np.asarray(g, dtype=int)
r = np.asarray(r, dtype=int)


print("\nPCA on blue:")
b = pcab)

print("PCA on green:")
g = pca(g)

print("PCA on red:")
r = pca(r)

# Merge channels into image and write it.
outImage = cv2.merge((b,g,r))
outImage = cv2.flip(outImage, 0)   #Flip image to change from original somehow for testing.
cv2.imwrite('flipped.jpg', outImage)

