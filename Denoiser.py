import pandas as pd 
from sklearn.model_selection import train_test_split
from HelperFunctions import *
from DistilBertModel import DistilBertModel, save_model, tokenizer


# Load the datasets
# Note that mac users seperated with , automatically, for windows users we have to specify ; as the seperator
english_test_dataset_labelled = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/Giuseppe.csv')
english_test_dataset_labelled_2 = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/andrea.csv', sep= ';')
english_test_dataset_labelled_3 = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/giovanni.csv', sep= ';')

# Concatenate the datasets
english_test_dataset_labelled = pd.concat([english_test_dataset_labelled, english_test_dataset_labelled_2, english_test_dataset_labelled_3], ignore_index= True)

# Remove comments that are not labelled (have NaN value in column 'Label')
english_test_dataset_labelled = english_test_dataset_labelled.dropna(subset=['Label'])

# Do preprocessing steps
english_test_dataset_labelled['Comment'] = P_data_cleaning(english_test_dataset_labelled['Comment'], 'english', False)

# Seperate the two columns in the dataframe into 'comment' and 'label' in form of two lists
english_test_dataset_labelled_comments = english_test_dataset_labelled['Comment'].tolist()
english_test_dataset_labelled_labels = english_test_dataset_labelled['Label'].tolist()

# Now seperate into noise and no noise
english_test_dataset_labelled_comments_noise = []

# We will refer to no noise as 0 and noise as 1. Turn every entry in english_test_dataset_labelled_labels into 1 if it is 'N', else into 0.
english_test_dataset_labelled_labels = [1 if label == 'N' else 0 for label in english_test_dataset_labelled_labels]

# Check the amount of noisy and non-noisy comments
print('Amount of noisy comments:', english_test_dataset_labelled_labels.count(1))
print('Amount of non-noisy comments:', english_test_dataset_labelled_labels.count(0))


# For easy testing purposes, I'll just take as much noisy comments as non-noisy comments. 
# If we label more, we could think about stuff like data augmentation (synonym replacement, etc.)
# Else, oversampling the noisy comments would be a good idea. But care for overfitting.

# Seperate noisy and non-noisy comments
english_test_dataset_labelled_comments_no_noise = []
english_test_dataset_labelled_comments_noise = []

for i in range(len(english_test_dataset_labelled_labels)):
    if english_test_dataset_labelled_labels[i] == 0:
        english_test_dataset_labelled_comments_no_noise.append(english_test_dataset_labelled_comments[i])
    else:
        english_test_dataset_labelled_comments_noise.append(english_test_dataset_labelled_comments[i])


# Take the first len(english_test_dataset_labelled_comments_noise) comments from the noisy comments
english_test_dataset_labelled_comments_no_noise = english_test_dataset_labelled_comments_no_noise[:len(english_test_dataset_labelled_comments_noise)]

# Check if the amount of noisy and non-noisy comments is the same
print('Amount of noisy comments:', len(english_test_dataset_labelled_comments_noise))
print('Amount of non-noisy comments:', len(english_test_dataset_labelled_comments_no_noise))

# Combine the comments and labels into a single list
comments = english_test_dataset_labelled_comments_no_noise + english_test_dataset_labelled_comments_noise
labels = [0]*len(english_test_dataset_labelled_comments_no_noise) + [1]*len(english_test_dataset_labelled_comments_noise)

# Split the data into training and validation sets with stratification
train_comments, val_comments, train_labels, val_labels = train_test_split(
    comments, labels, test_size=0.2, random_state=42, stratify=labels
)

# Check the amount of training and validation comments
print("We have {} training comments".format(len(train_comments)))
print("We have {} validation comments".format(len(val_comments)))

# Check that the classes are evenly distributed across training and validation sets
print("Training set:")
print("Noisy comments:", train_labels.count(1))
print("Non noisy comments:", train_labels.count(0))
print("Validation set:")
print("Noisy comments:", val_labels.count(1))
print("Non noisy comments:", val_labels.count(0))


# Using our evened out dataset, we can start applying the model
model_trained, tokenizer_trained = DistilBertModel(train_comments, train_labels, val_comments, val_labels, batch_size_train = 16, batch_size_val = 16, num_labels = 2, epochs = 1, tokenizer = tokenizer)
save_model(model_trained, tokenizer_trained, "denoising_model_fine_tuned_distilbert_english")


