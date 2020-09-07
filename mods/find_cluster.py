from sklearn import cluster
import matplotlib.pyplot as plt


def find_cluster(data):
    ks = range(1, 10)
    inertias = []
    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = cluster.KMeans(n_clusters=k)

        # Fit model to samples
        model.fit(data)

        # Append the inertia to the list of inertias
        inertia = model.inertia_
        inertias.append(inertia)
    # Plot ks vs Inertias
    plt.plot(ks, inertias, '-o')
    plt.xlabel('number of clusters k')
    plt.ylabel('inertia')
    plt.xticks(ks)
    plt.show()
