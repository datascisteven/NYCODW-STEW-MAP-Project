import re
import sys
import nltk
sys.path.append("../py")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def num_of_words(df, col):
    """create functions to count number of words in tweet"""
    df['word_ct'] = df[col].apply(lambda x: len(str(x).split(" ")))

def num_of_chars(df, col):
    """function to count number of characters in a tweet"""
    df['char_ct'] = df[col].str.len()

def avg_word(sentence):
    words = str(sentence).split()
    return (sum(len(word) for word in words)/len(words))

def avg_word_length(df, col):
    """function to calculate average word length and then average word length per tweet"""
    df['avg_wrd'] = df[col].apply(lambda x: avg_word(x))

def hash_ct(df, col):
    """function to count number of hashtags per tweet"""
    df['hash_ct'] = df[col].apply(lambda x: len(re.split(r'#', str(x)))-1)

def collect_and_remove_users(df, col):
    df['retweets'] = df[col].apply(lambda x: re.findall(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', str(x))) # remove re-tweet
    df.retweets = df.retweets.apply(lambda x: str(x)[1:-1])
    df['callouts'] = df[col].apply(lambda x: re.findall(r'(@[A-Za-z0-9-_]+)', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'(@[A-Za-z0-9-_]+)', '', str(x))) # remove tweeted at
    df.callouts = df.callouts.apply(lambda x: str(x)[1:-1])

def collect_and_remove_charef(df, col):
    df['charref'] = df[col].apply(lambda x: re.findall(r'&[\S]+?;', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', '', str(x)))
    df.charref = df.charref.apply(lambda x: str(x)[1:-1])

def collect_and_remove_hashtags(df, col):
    df['hashtags'] = df[col].apply(lambda x: re.findall(r'(#[A-Za-z]+[A-Za-z0-9-_]+)', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'(#[A-Za-z]+[A-Za-z0-9-_]+)', ' ', str(x)))
    df.hashtags = df.hashtags.apply(lambda x: str(x)[1:-1])

def collect_and_remove_links(df, col):
    df['urllinks'] = df[col].apply(lambda x: re.findall(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+', '', str(x)))  # remove http links
    df['shortlinks'] = df[col].apply(lambda x: re.findall(r'(http://(bit\.ly|t\.co|lnkd\.in|tcrn\.ch)\S*)\b', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(http://(bit\.ly|t\.co|lnkd\.in|tcrn\.ch)\S*)\b', '', str(x)))  # remove bit.ly links    
    df.urllinks = df.urllinks.apply(lambda x: str(x)[1:-1]) # remove brackets around list
    df.shortlinks = df.shortlinks.apply(lambda x: str(x)[1:-1])


def remove_numerics(df, col):
    """function to remove numbers or words with digits"""
    df[col] = df[col].apply(lambda x: re.sub(r'\w*\d\w*', r'', str(x)))

def remove_whitespaces(df, col):
    """function to remove any double or more whitespaces to single and any leading and trailing whitespaces"""
    df[col] = df[col].apply(lambda x: re.sub(r'\s\s+', ' ', str(x))) 
    df[col] = df[col].apply(lambda x: re.sub(r'(\A\s+|\s+\Z)', '', str(x))) 

def remove_special_char(df, col):
    df[col] = df[col].apply(lambda x: re.sub(r'&[\S]+?;', '', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', r'', str(x)))
    df[col] = df[col].apply(lambda x: re.sub(r'#', ' ', str(x)))

def make_lowercase(df, col):
    df[col] = df[col].apply(lambda x: str(x).lower())

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    stop_words = stopwords.words('english')
    stop_words += ['atatat']
    word_tokens = word_tokenize(tweet)
    no_stop_words = [lemmatize(x) for x in word_tokens if x not in stop_words]
    return no_stop_words

def lemmatize(token):
    """Returns lemmatization of a token"""
    return WordNetLemmatizer().lemmatize(token, pos='v')

def tokenize(tweet):
    """Returns tokenized representation of words in lemma form excluding stopwords"""
    stop_words = stopwords.words('english')
    stop_words += ['atatat']
    word_tokens = word_tokenize(tweet)
    no_stop_words = [lemmatize(x) for x in word_tokens if x not in stop_words]
    return no_stop_words

def tokenize_and_lemmatize(df, col):
    df[col] = df[col].apply(lambda x: tokenize(x))



def preprocess_tweets(df, col):
    """master function to preprocess tweets"""
    num_of_words(df, col)
    num_of_chars(df, col)
    avg_word_length(df, col)
    hash_ct(df, col)
    collect_and_remove_users(df, col)
    collect_and_remove_charef(df, col)
    collect_and_remove_hashtags(df, col)
    collect_and_remove_links(df, col)
    remove_numerics(df, col)
    remove_whitespaces(df, col)
    remove_special_char(df, col)
    make_lowercase(df, col)
    tokenize_and_lemmatize(df, col)
    return df


def preprocess(tweet):
    result = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    result = re.sub(r'(@[A-Za-z0-9-_]+)', '', result)
    result = re.sub(r'http\S+', '', result)
    result = re.sub(r'bit.ly/\S+', '', result)
    # result = re.sub(r'(.)\1+', r'\1\1', result)
    result = " ".join(re.findall('[A-Z][^A-Z]*', result)) 
    result = re.sub(r'&[\S]+?;', '', result)
    result = re.sub(r'#', ' ', result)
    result = re.sub(r'[^\w\s]', r'', result)    
    result = re.sub(r'\w*\d\w*', r'', result)
    result = re.sub(r'\s\s+', ' ', result)
    result = re.sub(r'(\A\s+|\s+\Z)', '', result)
    result = tokenize(result)
    return list(result)





