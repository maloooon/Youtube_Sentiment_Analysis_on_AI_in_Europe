"""This file tests how good the distilbert-base-multilingual-cased-sentiment model is at predicting the sentiment of some of our data.
   To this effect. We have labelled around 6000 comments ourselves in order to check the accuracy of the model."""


import pandas as pd 
from sklearn.model_selection import train_test_split
from HelperFunctions import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline



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


english_test_dataset_labelled.to_csv("/Users/marlon/VS-Code-Projects/Youtube/ENGLISH_OWNLABEL_FULL.csv")
# Drop noisy comments, as they are not useful for training
english_test_dataset_labelled = english_test_dataset_labelled[english_test_dataset_labelled['Label'] != 'N']

english_test_dataset_labelled.to_csv("/Users/marlon/VS-Code-Projects/Youtube/ENGLISH_OWNLABEL_NONOISE.csv")

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



# Concatenate the comments
comments = negative_comments + neutral_comments + positive_comments

# Turn all elements in comments into strings
comments = [str(comment) for comment in comments]

# Check that all values in comments are strings
for comment in comments:
    assert type(comment) == str, "All comments should be strings."

# Create the labels for the evened out dataset
labels = ['negative']*len(negative_comments) + ['neutral']*len(neutral_comments) + ['positive']*len(positive_comments)

print("We have ", len(comments), " comments and ", len(labels), " labels.")


# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("philschmid/distilbert-base-multilingual-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("philschmid/distilbert-base-multilingual-cased-sentiment")


# Initialize the sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Predict sentiment labels
predictions = sentiment_analysis(comments)

# Extract the scores from the predictions
scores = [prediction['score'] for prediction in predictions]

# Extract the labels from the predictions
predictions = [prediction['label'] for prediction in predictions]


print(f"We got : {len(predictions)} and {len(labels)} many labels")

negative_correct = 0
neutral_correct = 0
positive_correct = 0
same_count = 0
for i in range(len(predictions)):
    if labels[i] == predictions[i]:
        if labels[i] == 'negative':
            negative_correct += 1
        elif labels[i] == 'neutral':
            neutral_correct += 1
        elif labels[i] == 'positive':
            positive_correct += 1

        same_count += 1

accuracy = same_count / len(predictions)



print("We get an accuracy on all comments of : {}".format(accuracy))

print("Out of ", len(negative_comments), " negative comments, we got ", negative_correct, " correct. Accuracy : ", negative_correct/len(negative_comments))
print("Out of ", len(neutral_comments), " neutral comments, we got ", neutral_correct, " correct. Accuracy : ", neutral_correct/len(neutral_comments))
print("Out of ", len(positive_comments), " positive comments, we got ", positive_correct, " correct. Accuracy : ", positive_correct/len(positive_comments))



# Check different confidence scores to see how the model performs
best_acc_neg = 0
best_acc_pos = 0
conf_scores_test = [0.97,0.98,0.985, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995]
for conf_score in conf_scores_test:
    high_confidence_predictions = []
    high_confidence_labels = []
    high_confidence_comments = []
    for i in range(len(scores)):
        if scores[i] >= conf_score:
            high_confidence_predictions.append(predictions[i])
            high_confidence_labels.append(labels[i])
            high_confidence_comments.append(comments[i])



    # Seperate comments into the three categories
    negative_comments_high_confidence = []
    neutral_comments_high_confidence = []
    positive_comments_high_confidence = []

    for i in range(len(high_confidence_labels)):
        if high_confidence_labels[i] == 'negative':
            negative_comments_high_confidence.append(high_confidence_comments[i])
        elif high_confidence_labels[i] == 'neutral':
            neutral_comments_high_confidence.append(high_confidence_comments[i])
        elif high_confidence_labels[i] == 'positive':
            positive_comments_high_confidence.append(high_confidence_comments[i])


    print(f"We got : {len(high_confidence_predictions)} and {len(high_confidence_labels)} many labels")

    negative_correct = 0
    neutral_correct = 0
    positive_correct = 0
    same_count = 0
    for i in range(len(high_confidence_predictions)):
        if high_confidence_labels[i] == high_confidence_predictions[i]:
            if high_confidence_labels[i] == 'negative':
                negative_correct += 1
            elif high_confidence_labels[i] == 'neutral':
                neutral_correct += 1
            elif high_confidence_labels[i] == 'positive':
                positive_correct += 1

            same_count += 1

    accuracy = same_count / len(high_confidence_predictions)



    print("We get an accuracy on the {} confidence comments of : {}".format(conf_score, accuracy))

    print("Out of ", len(negative_comments_high_confidence), conf_score, " confidence scored negative comments, we got ", negative_correct, " correct. Accuracy : ", negative_correct/len(negative_comments_high_confidence))
    print("Out of ", len(neutral_comments_high_confidence), conf_score, " confidence scored neutral comments, we got ", neutral_correct, " correct. Accuracy : ", neutral_correct/len(neutral_comments_high_confidence))
    print("Out of ", len(positive_comments_high_confidence), conf_score,  " confidence scored positive comments, we got ", positive_correct, " correct. Accuracy : ", positive_correct/len(positive_comments_high_confidence))

    # Save the current best accuracy and compare to the one from the previous iteration
    if negative_correct/len(negative_comments_high_confidence) > best_acc_neg:
        best_acc_neg = negative_correct/len(negative_comments_high_confidence)
        # Also store the comments (also the ones that were classified wrong, as they still have a high confidence score : we might relabel !)
        negative_comments_high_confidence_best = negative_comments_high_confidence
        best_neg_predictions = high_confidence_predictions
        best_neg_labels = high_confidence_labels
    if positive_correct/len(positive_comments_high_confidence) > best_acc_pos:
        best_acc_pos = positive_correct/len(positive_comments_high_confidence)
        positive_comments_high_confidence_best = positive_comments_high_confidence
        best_pos_predictions = high_confidence_predictions
        best_pos_labels = high_confidence_labels




# Save the best comments with our labels (column named 'labels') and the predicted labels (column named 'predictions')
negative_comments_high_confidence_best = pd.DataFrame(negative_comments_high_confidence_best, columns=['comments'])
negative_comments_high_confidence_best['labels'] = 'negative'
predictions_on_negative = [best_neg_predictions[i] for i in range(len(best_neg_labels)) if best_neg_labels[i] == 'negative']

negative_comments_high_confidence_best['predictions'] = predictions_on_negative

positive_comments_high_confidence_best = pd.DataFrame(positive_comments_high_confidence_best, columns=['comments'])
positive_comments_high_confidence_best['labels'] = 'positive'
predictions_on_positive = [best_pos_predictions[i] for i in range(len(best_pos_labels)) if best_pos_labels[i] == 'positive']

positive_comments_high_confidence_best['predictions'] = predictions_on_positive

negative_comments_high_confidence_best.to_csv("/Users/marlon/VS-Code-Projects/Youtube/negative_comments_high_confidence_best.csv")
positive_comments_high_confidence_best.to_csv("/Users/marlon/VS-Code-Projects/Youtube/positive_comments_high_confidence_best.csv")



