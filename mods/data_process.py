import pandas as pd


# # loads the data set by pandas
def init(path):
    # loads the data set by pandas
    dataset = pd.read_csv(path, sep='\s+', header=None)
    # add column name to every column
    dataset.columns = ["area", "perimeter", "compactnes", "length of kernel", "width of kernel",
                       "asymmetry coefficient",
                       " length of kernel groove", "type"]
    return dataset


# create seeds list for cross-tabulation
def create_seeds_list(seed_type):
    seeds_list = []
    for v in seed_type.values:
        if v == 1:
            seeds_list.append("Canadian wheat")
        elif v == 2:
            seeds_list.append("Karma wheat")
        else:
            seeds_list.append("Rosa wheat")
    return seeds_list


# extract only the seeds data from original dataset remove target column
def get_data(dataset):
    data = dataset[["area", "perimeter", "compactnes", "length of kernel", "width of kernel", "asymmetry coefficient",
                    " length of kernel groove"]]
    return data
