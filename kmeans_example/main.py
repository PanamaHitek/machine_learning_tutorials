import matplotlib
from keras.datasets import mnist
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from sklearn import metrics

def infer_cluster_labels(kmeans, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """

    inferred_labels = {}

    for i in range(kmeans.n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(kmeans.labels_ == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if np.argmax(counts) in inferred_labels:
            # append the new number to the existing array at this slot
            inferred_labels[np.argmax(counts)].append(i)
        else:
            # create a new array in this slot
            inferred_labels[np.argmax(counts)] = [i]

        # print(labels)
        # print('Cluster: {}, label: {}'.format(i, np.argmax(counts)))

    return inferred_labels


def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """

    # empty array of len(X)
    predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key

    return predicted_labels

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    X = x_train.reshape(len(x_train),-1)
    Y = y_train

    X = X.astype(float) / 255.
    n_digits = len(np.unique(y_test))

    # Initialize and fit KMeans algorithm
    kmeans = MiniBatchKMeans(n_clusters=20)
    kmeans.fit(X)

    # record centroid values
    centroids = kmeans.cluster_centers_

    # reshape centroids into images
    images = centroids.reshape(20, 28, 28)
    images *= 255
    images = images.astype(np.uint8)

    # determine cluster labels
    cluster_labels = infer_cluster_labels(kmeans, Y)

    # create figure with subplots using matplotlib.pyplot
    fig, axs = plt.subplots(3, 6, figsize=(10, 10))
    plt.gray()

    # loop through subplots and add centroid images
    for i, ax in enumerate(axs.flat):

        # determine inferred label using cluster_labels dictionary
        for key, value in cluster_labels.items():
            if i in value:
                ax.set_title('Inferred Label: {}'.format(key),y=100)

        # add image to subplot
        ax.matshow(images[i],interpolation="none")
        ax.axis('off')

    plt.show()

if __name__ == "__main__":
    main()


