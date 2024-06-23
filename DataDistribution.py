import pandas as pd
import numpy as np

# Load in pandas dataframe
comments_processed = pd.read_csv('comments_processed.csv')
comments_original = pd.read_csv('comments_original.csv')

# Then split the comments into 5 dataframes of the same amount of rows 
# (or as close as possible) and save them as csv files
comments_processed_split = np.array_split(comments_processed, 5)
comments_original_split = np.array_split(comments_original, 5)

for i in range(5):
    comments_processed_split[i].to_csv('comments_processed_split_' + str(i) + '.csv', index=False)
    comments_original_split[i].to_csv('comments_original_split_' + str(i) + '.csv', index=False)

# Now we have 5 csv files for each of the processed and original comments
