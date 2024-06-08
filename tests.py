
import glob 
import pandas as pd
# Checking if all the comments were loaded since google colab broke down
all_english_comments = glob.glob('/Users/marlon/VS-Code-Projects/Youtube/english_data/*.csv')
all_english_comments_labelled = pd.read_csv('/Users/marlon/VS-Code-Projects/Youtube/High_Confidence_Comments_English (2).csv')

# Read in the data
all_english_comments = pd.concat([pd.read_csv(f) for f in all_english_comments], ignore_index = True)


# Remove comments with words like "video" and "channel" as they are associated with comments such as 'great video!'
all_english_comments = all_english_comments[~all_english_comments['Comment'].str.contains('video|channel', case=False)]

# Remove comments whose length is less than 3 words
all_english_comments = all_english_comments[all_english_comments['Comment'].str.split().str.len() > 3]

# Now we prepare for the labelling phase using a pre-trained state-of-the-art model

# Turn dataframe into a list
comments = all_english_comments['Comment'].tolist()
comments_labelled = all_english_comments_labelled['comments'].tolist()

# Turn all comments into strings
comments = [str(comment) for comment in comments]
comments_labelled = [str(comment) for comment in comments_labelled]

print(len(comments))
print(len(comments_labelled))

# TODO : we have a difference since google colab broke down, but it is minimal, so we can also just go with it