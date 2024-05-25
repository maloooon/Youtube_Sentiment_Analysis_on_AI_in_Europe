import regex as re
from nltk.corpus import stopwords

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



def P_data_cleaning(data, language, labelling):
    """Function to clean our data.
       data : data input ; pd.Series
       language : what language the comments are in (input in lowercase) : str
       labelling : if we want to label, we keep punctuation & stopwords
    """

    # REMOVE NAN ENTRIES
    data = data.dropna()

    # REMOVE COMMENTS THAT EXCEED CERTAIN LENGTH (350 for now)
    data = data[data.str.len() <= 350]
    

    # FOR GERMAN DATA : Change ö , ä , ü to oe, ae, ue
    data = data.str.replace("ö", "oe").str.replace("ä", "ae").str.replace("ü", "ue")

    # REMOVE NAMES FROM ANSWERS (in youtube comments scraper answers stored by @@)
    data = data.str.replace('@@\w+', '', regex=True)

    # REMOVING PUNCTUATION
    if labelling == False:
      data = data.str.replace('[^a-zA-Z0-9]',' ')

    # REMOVING EMOJIS
    data = data.apply(lambda x: remove_emoji(x))

    # LOWERCASE
    data = data.str.lower()

    # REMOVING STOPWORDS
    if labelling == False:
      data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words(language))]))


    return data
