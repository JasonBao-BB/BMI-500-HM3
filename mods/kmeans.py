from sklearn import cluster
import matplotlib.pyplot as plt


def k_means(seeds_data):
    # Create a KMeans model with 3 clusters model
    model = cluster.KMeans(n_clusters=3)
    # Use fit_predict to fit model and obtain cluster labels
    labels = model.fit_predict(seeds_data)
    # print(labels)

    # area as x-coordinate, perimeter as y-coordinate
    xs = seeds_data['area']
    ys = seeds_data['perimeter']
    # plot the graph with label we predict
    plt.xlabel('area A')
    plt.ylabel('perimeter P')
    plt.scatter(xs, ys, c=labels)

    # Assign the cluster centers: centroids
    centroids = model.cluster_centers_
    # Assign the columns of centroids: centroids_x, centroids_y
    centroids_x = centroids[:, 0]
    centroids_y = centroids[:, 1]
    # Make a scatter plot of centroids_x and centroids_y
    plt.scatter(centroids_x, centroids_y, marker='D', s=50)
    plt.show()
    return labels
