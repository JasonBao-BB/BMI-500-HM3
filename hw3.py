##################################################################################################
# Author Boyang Bao Date 09/05/2020
# This is BMI 500 Homework for Week 3
##################################################################################################
# Data comes from University of California Irvine
# UCI Machine Learning Repository
# Seeds Dataset can be download from this the link
# https://archive.ics.uci.edu/ml/datasets/seeds#
##################################################################################################

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster

# loads the data set by pandas
dataset = pd.read_csv('seeds_dataset.txt', sep='\s+', header=None)
# add column name to every column
dataset.columns = ["area", "perimeter", "compactnes", "length of kernel", "width of kernel", "asymmetry coefficient",
                   " length of kernel groove", "type"]

# create list of seeds
seed_type = pd.Series(dataset["type"])
seed_category = []
for v in seed_type.values:
    if v == 1:
        seed_category.append("Canadian wheat")
    elif v == 2:
        seed_category.append("Karma wheat")
    else:
        seed_category.append("Rosa wheat")

# extract the data from original data set
data = dataset[["area", "perimeter", "compactnes", "length of kernel", "width of kernel", "asymmetry coefficient", " length of kernel groove"]]

# find the best number of clusters for data prediction
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

# Create a KMeans model with 3 clusters model
model = cluster.KMeans(n_clusters=3)
# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(data)
# print(labels)

# area as x-coordinate, perimeter as y-coordinate
xs = data['area']
ys = data['perimeter']
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


# Create a DataFrame with labels and varieties as columns assigned to df
df = pd.DataFrame({'labels': labels, 'varieties': seed_category})
# print(df)
# Create cross-tabulation assigned to ct
ct = pd.crosstab(df['labels'], df['varieties'])
# Display cross-tabulation
# The cross-tabulation shows that the 3 varieties of grain separate really well into 3 clusters
print(ct)
