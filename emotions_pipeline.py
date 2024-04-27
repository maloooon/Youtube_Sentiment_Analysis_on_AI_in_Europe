import pandas as pd


set_training = pd.read_csv('training_dataset_v0/go_emotions_dataset.csv')

print(set_training.head())

# Note that it is possible that we have rows with no given emotion (e.g. for sentences with no clear meaning/ hard to read emotions from)
#print(set_training.iloc[1,3:])
