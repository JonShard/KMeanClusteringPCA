import numpy as np
import random as rng
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iterations = 10

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
            if count > 0:
                averageX /= count
                averageY /= count
                centroidsX[i] = averageX
                centroidsY[i] = averageY
        print("Finished iteration ", it+1)

    distanceToNearest = []
    for i in range(0, n):
        for j in range(0, len(x)):
            distanceToNearest.append(np.linalg.norm([x[j] - centroidsX[int(closestCentroid[i])],  y[j] - centroidsY[int(closestCentroid[i])]]))
    inertia = np.sum(distanceToNearest)
    print("Done. Inertia: ", inertia)
    return closestCentroid, inertia, centroidsX, centroidsY


def plotClasses(x, y, centX, centY, n, classType, name):
    colors = [('m'), ('b'), ('r'), ('g'), ('k'), ('y')]
    for i in range(0, n):
        centroid0Y = []
        centroid0X = []
        for j in range(0, len(x)):
            if int(classType[j]) == i:
                centroid0X.append(x[j])
                centroid0Y.append(y[j])
        plt.scatter(centroid0X, centroid0Y, c=colors[i])

    plt.scatter(centX, centY, marker='X')
    plt.savefig(name + "jpg")
    plt.show()
    plt.clf()


# Good example here: https://pythonprogramminglanguage.com/kmeans-elbow-method/
def elbowCluserCount(x, y, name):
    optimalClusters = 3

    kValues = range(1, 10)
    distortions = []
    for k in kValues:
        _, inertia, _, _ = kClustering(x, y, k, iterations)
        distortions.append(inertia)

    plt.plot(kValues, distortions)
    plt.savefig(name + ".jpg")
    plt.show()
    return optimalClusters


# Synthetic data set:
# Load data and save a figure of it..
data = np.loadtxt(open("cluster.txt", "r"), delimiter=",")
x = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
plt.scatter(x, y)
plt.savefig("syntheticPoints.jpg")
plt.show()
plt.clf()
clusterCountSynth = elbowCluserCount(x, y, "SynthElbow")
classTypesSynth, _, centXSynth, centYSynth = kClustering(x, y, clusterCountSynth, iterations)
plotClasses(x, y, centXSynth, centYSynth, clusterCountSynth, classTypesSynth, "SyntheticCentroids")


# Iris data set:
iris=load_iris()
# Pick 2 columns from the set for X and Y:
irisX = iris.data[:, 0]
irisY = iris.data[:, 2]
plt.scatter(irisX, irisY)
plt.savefig("irisPoints.jpg")
plt.show()
plt.clf()
clusterCountIris = elbowCluserCount(irisX, irisY, "IrisElbow")
classTypesIris, _, centXIris, centYIris = kClustering(irisX, irisY, clusterCountIris, iterations)

plotClasses(irisX, irisY, centXIris, centYIris, clusterCountIris, classTypesIris, "IrisCentroids")

