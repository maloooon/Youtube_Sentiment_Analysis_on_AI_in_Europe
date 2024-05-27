import pandas as pd 
from sklearn.model_selection import train_test_split
from HelperFunctions import *
from DistilBertModel import DistilBertModel, save_model, tokenizer



# Load the datasets
# Note that mac users seperated with , automatically, for windows users we have to specify ; as the seperator
english_test_dataset_labelled = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/Giuseppe.csv')
english_test_dataset_labelled_2 = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/andrea.csv', sep= ';')
english_test_dataset_labelled_3 = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/NLP labelled data preview/english set/giovanni.csv', sep= ';')

# Since the ones downloaded from windows are a bit messed up, we need to rename columns
english_test_dataset_labelled_2.columns = ["Comment", "Label"]
english_test_dataset_labelled_3.columns = ["Comment", "Label", "unnamed", "unnamed2"]

# Concatenate the datasets
english_test_dataset_labelled = pd.concat([english_test_dataset_labelled, english_test_dataset_labelled_2, english_test_dataset_labelled_3], ignore_index= True)

# Convert type of each comment to string
english_test_dataset_labelled['Comment'] = english_test_dataset_labelled['Comment'].astype(str)

# Remove the last 2 columns
english_test_dataset_labelled = english_test_dataset_labelled.drop(columns=['unnamed', 'unnamed2'])

# Remove comments that are not labelled (have NaN value in column 'Label')
english_test_dataset_labelled = english_test_dataset_labelled.dropna(subset=['Label'])

# Drop noisy comments, as they are not useful for training
english_test_dataset_labelled = english_test_dataset_labelled[english_test_dataset_labelled['Label'] != 'N']

# Do preprocessing steps
english_test_dataset_labelled['Comment'] = P_data_cleaning(english_test_dataset_labelled['Comment'], 'english', False)

# Seperate the two columns in the dataframe into 'comment' and 'label' in form of two lists
english_test_dataset_labelled_comments = english_test_dataset_labelled['Comment'].tolist()
english_test_dataset_labelled_labels = english_test_dataset_labelled['Label'].tolist()


# Convert the elements in the list to integers, handling non-integer values 
for idx,label in enumerate(english_test_dataset_labelled_labels):
    try:
        english_test_dataset_labelled_labels[idx] = int(label)
    except ValueError:
        # Handle the case where the label is not an integer
        english_test_dataset_labelled_labels[idx] = None


# Assess how many labelled comments we have
print("We have ", len(english_test_dataset_labelled_comments), " labelled comments in our dataset.")

# Check how many negative (-1), neutral (0) and positive (1) comments we have
print("We have ", english_test_dataset_labelled_labels.count(-1), " negative comments.")
print("We have ", english_test_dataset_labelled_labels.count(0), " neutral comments.")
print("We have ", english_test_dataset_labelled_labels.count(1), " positive comments.")

# Assert that the number of comments and labels are the same
assert len(english_test_dataset_labelled_comments) == len(english_test_dataset_labelled_labels), "The number of comments and labels are not the same."

# Even out the datasets

# Find the minimum number of comments in a category
min_comments = min(english_test_dataset_labelled_labels.count(-1), english_test_dataset_labelled_labels.count(0), english_test_dataset_labelled_labels.count(1))

# Seperate the 3 categories
negative_comments = []
neutral_comments = []
positive_comments = []

for i in range(len(english_test_dataset_labelled_labels)):
    if english_test_dataset_labelled_labels[i] == -1:
        negative_comments.append(english_test_dataset_labelled_comments[i])
    elif english_test_dataset_labelled_labels[i] == 0:
        neutral_comments.append(english_test_dataset_labelled_comments[i])
    elif english_test_dataset_labelled_labels[i] == 1:
        positive_comments.append(english_test_dataset_labelled_comments[i])

negative_comments = negative_comments[:min_comments]
neutral_comments = neutral_comments[:min_comments]
positive_comments = positive_comments[:min_comments]

# Now that we have evened out the dataset, we can concatenate the lists
comments = negative_comments + neutral_comments + positive_comments

# Turn all elements in comments into strings
comments = [str(comment) for comment in comments]

# Check that all values in comments are strings
for comment in comments:
    assert type(comment) == str, "All comments should be strings."

# Create the labels for the evened out dataset
labels = [0]*min_comments + [1]*min_comments + [2]*min_comments



# Split the data into training and validation sets with stratification
train_comments, val_comments, train_labels, val_labels = train_test_split(
    comments, labels, test_size=0.2, random_state=42, stratify=labels
)


print("We have {} training comments".format(len(train_comments)))
print("We have {} validation comments".format(len(val_comments)))

# Check that the classes are evenly distributed across training and validation sets
print("Training set:")
print("Negative comments:", train_labels.count(0))
print("Neutral comments:", train_labels.count(1))
print("Positive comments:", train_labels.count(2))
print("Validation set:")
print("Negative comments:", val_labels.count(0))
print("Neutral comments:", val_labels.count(1))
print("Positive comments:", val_labels.count(2))


# Using our evened out dataset, we can start applying the model
model_trained, tokenizer_trained = DistilBertModel(train_comments, train_labels, val_comments, val_labels, batch_size_train = 16, batch_size_val = 16, num_labels = 3, epochs = 1, tokenizer = tokenizer)
save_model(model_trained, tokenizer_trained, "sentiment_model_fine_tuned_distilbert_english")



