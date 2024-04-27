import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import regex as re 
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')






def remove_emoji(comment):
    """Function to remove emojis.
        comment : data input ; str
        Taken from :
        https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    
    """

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', comment)


def P_data_reading(path):
    """Simple function to read in the data we want to use.
       path : the path pointing to our data ; csv file 
    """

    comments_data = pd.read_csv(path)

    return comments_data['Comment']



def P_data_cleaning(data, language):
    """Function to clean our data.
       data : data input ; pd.DataFrame
       language : what language the comments are in (input in lowercase) : str 
    """

    

    # REMOVING PUNCTUATION
    data = data.str.replace('[^a-zA-Z0-9]',' ')

    # REMOVING EMOJIS 
    data = data.apply(lambda x: remove_emoji(x))

    # LOWERCASE
    data = data.str.lower()

    # REMOVING STOPWORDS
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words(language))]))

    return data



def P_data_lemmatizing(data):
    """FILTER APPROACH 1 : We first lemmatize so we get the base words of everything
       + we have less words in general and can build bigger groups
       - we will lose some accuracy in our sentiment analysis : words like best/better/good will all be just good
       
    """

    pass



# LEMMATIZE OR TOKENIZE BEFORE THIS STEP : 

def P_data_pre_filtering(data):
    """FILTER APPROACH 1 : We find the buzz words we want to filter for.
       The idea is to iterate through our own data and see if there are
       some really common words that are used for showing ones sentiment
       if there is a pattern, we can use these to remove the noise from
       our data
       data : data input : pd.DataFrame
    """

    pass


def P_data_filtering(data, sentiment_words):
    """FILTERING APPROACH 1 : We do pre-filtering on our data to remove noise.

    """

    pass






def main():
    path = 'comments_0.csv'

    data = P_data_reading(path)
    data_cleaned = P_data_cleaning(data, language='german')





if __name__ == '__main__':
    main()


