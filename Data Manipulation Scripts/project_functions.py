import pandas as pd
import numpy as np

def binary_sentiment(non_bin_sent):
    """
    Creates a binarised sentiment score from the polarity scores given by VADER.
    Expects a list as input.
    """
    
    bin_sent = []
    
    for i in range(len(non_bin_sent)):
        if non_bin_sent[i] < 0:
            bin_sent.append(-1)
        else:
            bin_sent.append(1)
    
    return bin_sent

##Text Pre-Processing##
#Required modules
import spacy

#Lemmatization w/spacy
def space(comment):
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    doc = nlp(comment)
    return " ".join([token.lemma_ for token in doc])

#Required modules
import re
import pandas as pd
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

#https://towardsdatascience.com/cleaning-preprocessing-text-data-for-sentiment-analysis-382a41f150d6

def pre_process(data):
    """
    Implements relevant text pre-processing for posts/comments,
    used prior to conducting sentiment analysis.
    Expects a dataframe as input.
    """
    #Making all text lowercase
    #data_clean = data.apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    #Removing punctuation
    #data_clean = data_clean.str.replace('[^\w\s]','')
    data_clean = data.str.replace('[^\w\s]','')
    
    #Removing stopwords
    stop = stopwords.words('english')
    data_clean = data_clean.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    
    #Spacy Lemmatization
    data_clean = data_clean.apply(space)
    
    return data_clean

##VADER Sentiment##
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def vader_score(data):
    """
    Generates sentiment scores for a given set of text data (posts/comments)
    using VADERsentiment.
    Expects a dataframe column as input.
    """
    
    analyser = SentimentIntensityAnalyzer()
    
    senti_score = []
    
    for i in data:
        senti_score.append(list(analyser.polarity_scores(i).values())[-1])
        
    return senti_score

##Epoch Time to Conventional##
def epoch_time_convert(data):
    
    data_converted = pd.to_datetime(data,unit='s',dayfirst=True)
    
    for i in data_converted:
        i = str(i)
    
    return data_converted

def date_shorten(data):
    
    data = data.astype({"created":str},errors='raise')
    
    for i in range(len(data)):
        data['created'][i] = data['created'][i][:10]
        
    data = data.sort_values('created')
    
    return data

##Daily Sentiment##
def daily_sentiment(data, score):
    """
    Collates sentiment scores for posts made on the same day to create
    a sentiment score time series variable.
    Expects a dataframe for 'data' and the desired sentiment score metric 
    (e.g. VADER or textBlob) as 'score'
    """
    
    senti_time_series_df = data[['created',score]]
    
    senti_time_series_dict = {}

    for i in set(senti_time_series_df['created']):
        senti_time_series_dict[i] = 0
        
        
    for i in senti_time_series_dict.keys():
        for j in range(len(senti_time_series_df)):
            if i == senti_time_series_df.iloc[j,0]:
                senti_time_series_dict[i] += senti_time_series_df.iloc[j,1]
    
    return senti_time_series_dict

##Attention Score##
def post_attention_score(data):
    """
    Generates attention scores for a given set of text data (posts/comments)
    using VADERsentiment.
    Attention = number of mentions of 'inflation' or 'inflationary'/number of words in post.
    Expects a dataframe column as input.
    """
    attention_score = []
    
    for i in data:
        
        attention = 0

        if "inflation" in i.lower() or "inflationary" in i.lower():
            attention += 1
            
        attention_score.append(attention/len(i))
        
    return attention_score

def daily_attention(data, score = 'attention_score'):

    """
    Collates attention scores for posts made on the same day to create
    an attention score time series variable.
    Expects a dataframe for 'data' and the desired attention score metric column name
    (defaults to 'attention_score') as 'score'.
    """
    
    attention_time_series_df = data[['created',score]]
    
    attention_time_series_dict = {}

    for i in set(attention_time_series_df['created']):
        attention_time_series_dict[i] = 0
        
        
    for i in attention_time_series_dict.keys():
        for j in range(len(attention_time_series_df)):
            if i == attention_time_series_df.iloc[j,0]:
                attention_time_series_dict[i] += attention_time_series_df.iloc[j,1]
    
    return attention_time_series_dict

##Create Wordclouds#
from wordcloud import WordCloud, STOPWORDS

def get_wordcloud(column):
    """
    Generates a wordcloud for a given text input.
    Expects a dataframe column as input.
    """
    
    import matplotlib.pyplot as plt

    stopwords = set(STOPWORDS)
    comment_words = ''
    
    for post in column:
        tokens = post.split()

        for i in range(len(tokens)):
            tokens[i] = tokens[i].lower()

    comment_words += " ".join(tokens)+" "
    
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)
    
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()

##Sarcasm Checker##
def sarcasm_checker(df,score):
    """
    Acts as a sarcasm detection tool based on certain tests (e.g. punctuation, capitalisation).
    Expects a dataframe and relvant sentiment score metric (column name) as input.
    NB To be used *after* doing sentiment analysis with VADER.
    """
    #Conditions for sarcasm
    for i in df['body']:
        #Exclamation mark count
        if i.count("!") > 3:
            df['sarcasm'] = True

        caps_count = 0

        for letter in i:
            if letter.isupper() == True:
                caps_count += 1

            if caps_count > len(i.split()):
                df['sarcasm'] = True
            
        else:
            df['sarcasm'] = False
    
    #Final - sarcasm indicator
    for sarc in df['sarcasm']:
        if sarc == True and df[score] > 0:
            df['sentiment_final'] = df[score] * -1
        else:
            df['sentiment_final'] = df[score]
            
    return df 