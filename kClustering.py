import numpy as np
import random as rng
import matplotlib.pyplot as plt



def kClustering(x, y, n, iterations):
    print("Doing K clustering. Classes: %d, number of points: %d" % (n, len(x)))
    
    # Initiallize randomized centtroid points 
    centroidsX = []
    centroidsY = []
    
    for i in range(0,n):
        point = rng.randrange(0, len(x))
        centroidsX.append(x[point])
        centroidsY.append(y[point])

    for it in range(0, iterations):
        closestCentroid = np.empty(len(x))
        # Assign each point to one of the centroids based on distance:
        for i in range(0, len(x)):
            sqareDistances = []
            for j in range(0, n):
                sqareDistances.append(np.square(x[i] - centroidsX[j]) + np.square(y[i] - centroidsY[j]))
            
            smallest = np.min(sqareDistances)
            index = sqareDistances.index(smallest)
            closestCentroid[i] = index

        # Calculate the mean for each cluster, and make that point its new position:
        for i in range(0, n):
            averageX = 0
            averageY = 0
            count = 0
            for j in range(0, len(x)):
                if int(closestCentroid[j]) == i:
                    averageX += x[j]
                    averageY += y[j]
                    count += 1
            averageX /= count
            averageY /= count
            centroidsX[i] = averageX
            centroidsY[i] = averageY

        centroid0X = []
        centroid0Y = []
        centroid1X = []
        centroid1Y = []
        for i in range(0, len(x)):
            if int(closestCentroid[i]) == 0:
                centroid0X.append(x[i])
                centroid0Y.append(y[i])
            if int(closestCentroid[i]) == 1:
                centroid1X.append(x[i])
                centroid1Y.append(y[i])
        
        plt.scatter(centroid0X, centroid0Y, c='b')
        plt.scatter(centroid1X, centroid1Y, c='m')
        plt.scatter(centroidsX, centroidsY, marker='X')
        plt.savefig("centroids.jpg")
        plt.show()
        plt.clf()
        print("finished iteration ", it+1)

# Load data and save a figure of it..
data = np.loadtxt(open("cluster.txt", "r"), delimiter=",")
x = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
plt.scatter(x, y)
plt.savefig("points.jpg")
plt.show()

kClustering(x, y, 2, 10)
