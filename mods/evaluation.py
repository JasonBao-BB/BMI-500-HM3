import pandas as pd


def evaluate(labels, seeds_category):
    df = pd.DataFrame({'labels': labels, 'varieties': seeds_category})
    # print(df)
    # Create cross-tabulation assigned to ct
    ct = pd.crosstab(df['labels'], df['varieties'])
    # Display cross-tabulation
    # The cross-tabulation shows that the 3 varieties of grain separate really well into 3 clusters
    print(ct)