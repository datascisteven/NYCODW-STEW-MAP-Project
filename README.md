# NYCODW-STEW-MAP-Project
 
## Notebooks
1. Preprocess_Tweets.ipynb:  EDA and Preprocessing Tweets
    - EDA:
      - Word Count
      - Character Count
      - Average Word Length
      - Number of Hashtags
    - Preprocessing:
      - collect and remove:
        - @usernames
        - character references
        - #hashtags
        - url links
      - remove:
        - numerics
        - whitespaces
        - special characters
      - make lowercase
      - lemmatize
      - tokenize
    - Further EDA:
      - Frequency Distribution of Top 25 Tokens
      - WordCloud
2. Unsupervised_LDA.ipynb:  Extract Topics from Tweets
    - n_components: 3-6, 10
    - pyLDAvis
3. Training_LDA.ipynb:  Training LDA Models
    - Bag of Words
    - TD-IDF
    - Hyperparameter Tuning
      - have not finished yet

## Potential Models:

1. Training LDA with Wikipedia Corpus
    - Unpacking the corpus took a long time
    - Never fully explored this avenue

- Based on Jónsson, Elı́as. “An Evaluation of Topic Modelling Techniques for Twitter.” (2016):

2. Biterm Model
    - find a Python implementation of model
3. Word2vec vectors using a Gaussian mixture model - lda2vec:
    - https://towardsdatascience.com/combing-lda-and-word-embeddings-for-topic-modeling-fe4a1315a5b4


