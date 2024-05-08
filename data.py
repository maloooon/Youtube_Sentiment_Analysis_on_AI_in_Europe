import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
from nltk import ne_chunk
import regex as re 
import spacy # We need spacy for german lemmatization 
import de_core_news_sm
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
from transformers import pipeline
from transformers import DistilBertTokenizer
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
from germansentiment import SentimentModel








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


def P_data_reading(path, i = 20):
    """Simple function to read in the data we want to use.
       path : the path pointing to our data ; csv file 
    """

    comments_data = pd.read_csv(path)

    ############### FOR TESTING PURPOSES, WE ONLY TAKE FIRST 20 ###############

    # Turn into Series, containing only the comments
    return comments_data.iloc[0:i,:]['Comment']



def P_data_cleaning(data, language):
    """Function to clean our data.
       data : data input ; pd.Series
       language : what language the comments are in (input in lowercase) : str 
    """

    
    # FOR GERMAN DATA : Change ö , ä , ü to oe, ae, ue 
    data = data.str.replace("ö", "oe").str.replace("ä", "ae").str.replace("ü", "ue")

    # REMOVING PUNCTUATION
    data = data.str.replace('[^a-zA-Z0-9]',' ')

    # REMOVING EMOJIS 
    data = data.apply(lambda x: remove_emoji(x))

    # LOWERCASE
    data = data.str.lower()

    # REMOVING STOPWORDS
    data = data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords.words(language))]))



    return data


def P_data_tokenization(comment, language, model):
    """
    Tokenization function. We implement different tokenizers
    comment : the current comment to analyze ; string
    language : the language for tokenization ; string
    model : the tokenizer we are using (or from which model we are using the tokenizer from)
    """

    if model.lower() == 'distilbert':
        # We use the distilBERT tokenization (in case we are going to use that model later on)
        # NOTE : don't know what languages are included in multilingual, I just know german is in it 
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

      #  encoded_comment = tokenizer.encode(comment, add_special_tokens=True)


        # Tokenizing and padding comments (padding needed for constant input later on in distilBERT model)
        tokenized_comment = tokenizer.encode_plus(
            comment,
            max_length=128,  # Set the desired maximum sequence length
            padding='longest',  # Pad to the longest sequence in the batch
            truncation=True,  # Truncate if needed
            return_tensors='pt',  # Return PyTorch tensors
            )

        # Access the input IDs (we'll use these for fine-tuning (? on which data will we do fine-tuning ? Daniele proposed
        # the english comment section for AI on youtube, because there we have so much data and it is similar to ours))
        #input_ids = tokenized_comment['input_ids']
  
        # Will return input ids and attention mask of our inputs 
        return tokenized_comment





def P_data_lemmatizing(comment, language):
    """FILTER APPROACH 1 : We first lemmatize so we get the base words of everything
       + we have less words in general and can build bigger groups
       - we will lose some accuracy in our sentiment analysis : words like best/better/good will all be just good

       To combat the negative effect, we will do the following : Build a mapping between the original input sentences
       and the lemmatized ones. We will just lemmatize to build the bigger groups and denoise our dataset. Then, when we 
       done this, we map back to the original sentences and tokenize.

       Since we use pandas, we just won't reset indices. That way, we just keep the original pandas dataset (i.e. we save 
       a copy of it after the cleaning steps and right before lemmatizing) and then use the indices for our mapping.
       
       comment : the current comment to analyze ; string
       language : the language for tokenization ; string
    """
    if language.lower() == 'german':
        lemmatizer = spacy.load("de_core_news_sm")



    lemmatized_comment = ' '.join([token.lemma_ for token in lemmatizer(comment)])

    # After lemmatizing, some words are again higher cased 
    lemmatized_comment = lemmatized_comment.lower()


    return lemmatized_comment




# LEMMATIZE OR TOKENIZE BEFORE THIS STEP : 

def P_data_word_count(data):
    """FILTER APPROACH 1 : We find the buzz words we want to filter for.
       The idea is to iterate through our own data and see if there are
       some really common words that are used for showing ones sentiment
       if there is a pattern, we can use these to remove the noise from
       our data
       data : data input : pd.Series
    """
    # explode() : convert each single element into a row
    # We also sort them to find the most common ones
    word_counts = data.str.split().explode().value_counts().sort_values(ascending = False)



  

    # We return the count aswell as the (lemmatized) words themselves
    return word_counts, list(word_counts.index)


  



   


def P_data_filtering(sentiment_words, model, language, threshold = 0.95):
    """FILTERING APPROACH 1 : We do pre-filtering on our data to remove noise.
       For this, we use pre-trained, state-of-the-art models to find the sentiments of different words in different languages.
       Next, we filter the data (see details below)
       sentiment_words = list of words we want to use for filtering : List of String
       model : which model to use
       language : the language for tokenization ; string
       threshold : threshold on the confidence level of sentiment predictions of the single words ; Float
    """


    if language.lower() == 'german':
        if model.lower() == 'bert':
            model = SentimentModel() # Specifically trained on german texts ! 
            
            data = {'word' : [], 'sentiment_label' : [], 'confidence_pos' : [], 'confidence_neg' : [], 'confidence_neutral' : [], 'confidence_highest' : []}

            for word in sentiment_words:
                classes, probabilities = model.predict_sentiment([word], output_probabilities = True)
                data['word'].append(word)
                data['sentiment_label'].append(classes[0])
                data['confidence_pos'].append(probabilities[0][0][1])
                data['confidence_neg'].append(probabilities[0][1][1])
                data['confidence_neutral'].append(probabilities[0][2][1])
                data['confidence_highest'].append(max(probabilities[0][0][1],probabilities[0][1][1],probabilities[0][2][1]))
                
        
            words_sentiments_confidence = pd.DataFrame(data, columns=['word', 'sentiment_label', 'confidence_pos', 'confidence_neg', 'confidence_neutral', 'confidence_highest'])

            words_sentiments_confidence.to_csv('/Users/marlon/VS-Code-Projects/Youtube/words_and_sentiments.csv')

            # NOTE : I keep this in the german & bert loop since I don't know if we will have models for each language that output a 
            #        a confidence score
            # Next, based on some threshold, we only keep the words with positive / negative sentiment with a confidence >= threshold
            # Additionally, I found this pre-trained model to give numbers a positive sentiment with high confidence, so we remove these aswell
            # Also, sometimes it classifies a single letter with something positive/negative. Remove these aswell (in german, there are no single letter words)

        

            words_sentiments_confidence_filtered = words_sentiments_confidence[(words_sentiments_confidence['confidence_highest'] >= threshold)\
                                                                                & (words_sentiments_confidence['confidence_highest'] != words_sentiments_confidence['confidence_neutral']) \
                                                                                & (~words_sentiments_confidence['word'].str.contains(r'\d')) \
                                                                                & (words_sentiments_confidence['word'].str.len() > 1)]

            words_sentiments_confidence_filtered.to_csv('/Users/marlon/VS-Code-Projects/Youtube/words_and_sentiments_filtered.csv')


            # Finally, we look at the neutral values : Here, we use a list of buzz words that are AI related. We only want to keep
            # the neutral words that are somewhat related to AI.
            neutral_filter = ['ai', 'künstlich', 'künstliche', 'intelligenz', 'ki', 'machine', 'learning', 'kunst', 'roboter', 'robot']

            words_sentiments_confidence_filtered_2 = words_sentiments_confidence[(words_sentiments_confidence['word'].isin(neutral_filter))]

           # words_sentiments_confidence_filtered_2.to_csv('/Users/marlon/VS-Code-Projects/Youtube/words_and_sentiments_filtered_2.csv')

            words_sentiments_confidence_filtered_final = pd.concat([words_sentiments_confidence_filtered, words_sentiments_confidence_filtered_2])

            # Possible that we have some duplicates in the two concatenated ones (since in filtered_2 we take across also the ones with positive & negative sentiment again)
            words_sentiments_confidence_filtered_final = words_sentiments_confidence_filtered_final.drop_duplicates()

            words_sentiments_confidence_filtered_final.to_csv('/Users/marlon/VS-Code-Projects/Youtube/words_and_sentiments_filtered_final.csv')

            return words_sentiments_confidence_filtered_final


def P_data_remap(data_sentiments_filtered, data_lemmatized, data_only_cleaned):
    """
    FILTERING APPROACH 1: After we have found the words that show some strong sentiment or are connected to AI in some way,
    we now want to remap to the original sentences again 
    data_sentiments_filtered : the final words with all the different sentiments scores, filtered ; pd.DataFrame
    data_lemmatized : our lemmatized (and cleaned) words ; pd.Series
    data_only_cleaned : just cleaned data ; pd.Series
    """

    # We first create a list of all the words
    
    filtered_words = list(data_sentiments_filtered['word'])

    # Now we only want to keep the occurences where these words appear in our lemmatized version

    data_lemmatized_filtered = data_lemmatized[data_lemmatized.apply(lambda x: any(word in x for word in filtered_words))]


    data_lemmatized_filtered.to_csv('/Users/marlon/VS-Code-Projects/Youtube/test.csv')

    # And then finally we map back to the unlemmatized ones, because we will be using tokenization

   

    data_cleaned_and_filtered = data_only_cleaned[data_only_cleaned.index.isin(data_lemmatized_filtered.index)]

    data_cleaned_and_filtered.to_csv('/Users/marlon/VS-Code-Projects/Youtube/CLEANED_AND_FILTERED_APPROACH_1.csv')


    return data_cleaned_and_filtered 


    


def V_word_cloud(data):
    """ Visualization tool. A word cloud so we can see what words appears most.
        data : contains the counts of each word ; pd.Series
    """

    # Convert the series to a concatenated string
    comment_words = ' '.join([str(w) for w in data.index])

    # Generate the word cloud
    wordcloud = WordCloud(width=512, height=512, background_color='white', max_words=20).generate(comment_words)

    # Display the word cloud
    plt.figure(figsize=(10, 8), facecolor='white', edgecolor='blue')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()



def main():
    path = 'comments_0.csv'

    data = P_data_reading(path)
    data_cleaned = P_data_cleaning(data, language ='german')
    data_cleaned.to_csv('/Users/marlon/VS-Code-Projects/Youtube/check_cleaned.csv')
    data_cleaned_lemmatized = data_cleaned.apply(lambda x : P_data_lemmatizing(x, language = 'german'))
    data_cleaned_lemmatized.to_csv('/Users/marlon/VS-Code-Projects/Youtube/check_cleaned_lemmatized.csv')
    data_words_count, words = P_data_word_count(data_cleaned_lemmatized)
    words_sentiments_filtered = P_data_filtering(words, model= 'bert', language= 'german')
    data_cleaned_and_filtered = P_data_remap(words_sentiments_filtered, data_cleaned_lemmatized, data_cleaned)
    data_cleaned_and_filtered_tokenized = data_cleaned_and_filtered.apply(lambda x : P_data_tokenization(x, language= 'german', model = 'distilbert'))
    
    # With this final step, we can now build a dataloader and fine tune a distilBERT model
    # since we have unlabeled data, maybe something like self-supervised learning / transfer learning (with the english youtube AI comments,
    # like daniele proposed)


    data_cleaned_and_filtered_tokenized.to_csv('/Users/marlon/VS-Code-Projects/Youtube/cleaned_filtered_tokenized.csv')
   
    # Create the word cloud with the filtered data
    data_words_count, words = P_data_word_count(data_cleaned_and_filtered)
    V_word_cloud(data_words_count)









if __name__ == '__main__':
    main()


