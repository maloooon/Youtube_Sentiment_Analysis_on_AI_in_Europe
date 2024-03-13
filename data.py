import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk import ne_chunk
#nltk.download('punkt')
#nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

comments_data = pd.read_csv('comments_example.csv')


# Clean data 


# Remove punctuation
comments_data['comments'] = comments_data['comments'].str.replace('[^a-zA-Z0-9]',' ')

# Lowercase
comments_data['comments'] = comments_data['comments'].str.lower()

# Tokenization
comments_data['comments'] = comments_data['comments'].apply(word_tokenize).astype('object')

comments_data_tagged = comments_data['comments'].apply(pos_tag).tolist()



# We work on lists now (how to store lists as objects in pandas columns ??) bc no idea how to apply this function to a pd column the right way

# Stopwords
comments_data_list = comments_data['comments'].tolist()

for idx,sentence in enumerate(comments_data_list):
    comments_data_list[idx] = [w for w in sentence if w not in stopwords.words('english')]




# Find occurences of names
    















comments_data['comments'].to_csv('comments_check.csv')
