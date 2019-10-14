import numpy as np
import random as rng
import matplotlib.pyplot as plt

def kClustering(x, y, n):
    print("Doing K clustering. Classes: %d, number of points: %d" % (n, len(x)))
    
    # Initiallize randomized centtroid points 
    centroidsX = []
    centroidsY = []
    closestCentroid = np.empty(len(x))
    
    for i in range(0,n):
        centroidsX.append(rng.randrange(int(np.average(x) - np.average(x)/2), int(np.average(x) + np.average(x)/2)))
        centroidsY.append(rng.randrange(int(np.average(y) - np.average(y)/2), int(np.average(y) + np.average(y)/2)))
    
    for i in range(0, len(x)): # For each data point:
        sqareDistances = []
        for j in range(0, n):
            sqareDistances.append(np.square(x[i] - centroidsX[j]) + np.square(y[i] - centroidsY[j]))
        
        smallest = np.min(sqareDistances)
        index = sqareDistances.index(smallest)
        closestCentroid[i] = index

    plt.scatter(x, y)
    plt.scatter(centroidsX, centroidsY, marker='X')
    plt.savefig("centroids.jpg")


# Load data and save a figure of it..
data = np.loadtxt(open("cluster.txt", "r"), delimiter=",")
x = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
plt.scatter(x, y)
plt.savefig("points.jpg")


kClustering(x, y, 2)
