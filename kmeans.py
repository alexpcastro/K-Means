#!/usr/bin/python
#
# Author: Alexander Castro
# Description: A Python implementation of the k-means clustering algorithm that colors a given image.
#

import numpy
import skimage.io
import matplotlib.pyplot as plt

numpy.seterr(divide='ignore')

class KMeans:
    def fit(self,data):
        k = 10 # K-value of desired clusters
        starting_centroids = numpy.array([[0, 0, 0],[0.1, 0.1, 0.1],[0.2, 0.2, 0.2],[0.3, 0.3, 0.3],[0.4, 0.4, 0.4],[0.5, 0.5, 0.5],[0.6, 0.6, 0.6],[0.7, 0.7, 0.7],[0.8, 0.8, 0.8],[0.9, 0.9, 0.9]])

        self.centroids = {}
        for num in range(k):
            self.centroids[num] = starting_centroids[num]

        # Maximum number of iterations to perform if convergence doesn't occur
        for x in range(50):

            # Create k empty entries in the clusters dictionary
            self.clusters = {}
            for num in range(k):
                self.clusters[num] = []

            # For each pixel in our img, calculate the distance between that point and the centroid for each cluster. Use the index of the minumum distance to determine the corresponding cluster and add the pixel to that cluster.
            for pixel in data:
                distances = [numpy.linalg.norm(pixel-self.centroids[centroid]) for centroid in self.centroids]
                # Add pixel to the corresponding cluster using index of min. centroid distance
                self.clusters[distances.index(min(distances))].append(pixel)

            # Now that we have our new centroids for each cluster, save the old centroids for convergence comparison
            old_centroids = dict(self.centroids)

            # Selects each cluster and averages all the points within it and saves that point as the new centroid for that cluster
            for cluster in self.clusters:
                self.centroids[cluster] = numpy.average(self.clusters[cluster],axis=0)

            # Assume centroid move was optimized before testing new centroid against old centroid considering desired tolerance of convergence (did it move more than tolerance)
            convergence = True

            # Loop over every centroid and check the percentage amount that the old original_centroid has changed compared to the new current_centroid. For each centroid, if it has moved more than our tolerance 1.e-3, the centroid is not optimized and the program checks the next centroids.
            for c in self.centroids:
                if numpy.sum((self.centroids[c] - old_centroids[c] ) / old_centroids[c] * 100.0) > 1.e-3:
                    convergence = False

            # If the centroid is optimized (convergence occurs within tolerance), the centroid is optimized and we can stop.
            if convergence:
                break

original_img = skimage.io.imread('image.png') # Given by professor, turns image into numpy array
#skimage.io.imshow(original_img, vmin=0, vmax=255)
img = original_img.reshape((original_img.shape[0] * original_img.shape[1], 3))

img = img/255; # Normalize all values as requested

kmeans = KMeans()
kmeans.fit(img)

sum_squared_error = 0
total_clustered = 0

# Loop over each cluster created by the k-means algorithm and then loop over each point in every cluster. Here we will calculate the squared error of each point and it's centroid

for cluster in kmeans.clusters:

    # Display number of items in each cluster to
    cluster_size = len(kmeans.clusters[cluster])
    print("Items in cluster {}: {}".format(cluster,cluster_size))
    total_clustered += cluster_size
    for pixel in kmeans.clusters[cluster]:
        # Calculate squared error for this pixel
        pixel_squared_error = numpy.linalg.norm(pixel-kmeans.centroids[cluster])
        sum_squared_error += pixel_squared_error
        #print("Squared error: {}".format(pixel_squared_error))

        # As per instructions, after we've run k-means and clustered the data, change the rgb values of the data to the colors requested in instructions.
        if cluster == 0:
            pixel[0],pixel[1],pixel[2] = 60, 179, 113
        elif cluster == 1:
            pixel[0],pixel[1],pixel[2] = 0, 191, 255
        elif cluster == 2:
            pixel[0],pixel[1],pixel[2] = 255, 255, 0
        elif cluster == 3:
            pixel[0],pixel[1],pixel[2] = 255, 0, 0
        elif cluster == 4:
            pixel[0],pixel[1],pixel[2] = 0, 0, 0
        elif cluster == 5:
            pixel[0],pixel[1],pixel[2] = 169, 169, 169
        elif cluster == 6:
            pixel[0],pixel[1],pixel[2] = 255, 140, 0
        elif cluster == 7:
            pixel[0],pixel[1],pixel[2] = 128, 0, 128
        elif cluster == 8:
            pixel[0],pixel[1],pixel[2] = 255, 192, 203
        elif cluster == 9:
            pixel[0],pixel[1],pixel[2] = 255, 255, 255

# Print total pixels in all clusters, which should equal 198*244 = 48,312 for any k value
print("{} pixels in {} clusters.".format(total_clustered,len(kmeans.clusters)))

# Since we edited img, lets reshape it back to the original shape and load it into skimage so it loads it with the changes now.
img = img.reshape(original_img.shape[0],original_img.shape[1],3)
skimage.io.imshow(img, vmin=0, vmax=255)

plt.show()
print("Total SSE: {}".format(sum_squared_error))
