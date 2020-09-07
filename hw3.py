##################################################################################################
# Author Boyang Bao Date 09/05/2020
# This is BMI 500 Homework for Week 3
##################################################################################################
# Data comes from University of California Irvine
# UCI Machine Learning Repository
# Seeds Dataset can be download from this the link
# https://archive.ics.uci.edu/ml/datasets/seeds#
##################################################################################################
from mods import data_process as dp
from mods import find_cluster as fc
from mods import kmeans as km
from mods import evaluation

# # loads the data set by pandas
dataset = dp.init('./resource/seeds_dataset.txt')
# print(dataset)

# create seeds list for cross-tabulation
seeds_category = dp.create_seeds_list(dataset['type'])
# print(seeds_category)

# extract only the seeds data from original dataset remove target column
seeds_data = dp.get_data(dataset)
# print(seeds_data)

# find the best number of clusters for data prediction
fc.find_cluster(seeds_data)

# use k-means algorithm to find clusters
labels = km.k_means(seeds_data)

# test clusters accuracy
evaluation.evaluate(labels, seeds_category)
