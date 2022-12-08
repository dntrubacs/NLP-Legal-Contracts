"""Created by Daniel-Iosif Trubacs on 6 December 2022 for the AI society. The aim of this modules is
to process the cleaned dataset. For the word tokenization, stop words removal and stemming are used.
For the word encoding TF-IDF is used (see more here: https://en.wikipedia.org/wiki/Tf%E2%80%93idf).
"""

import string
import json
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# a function to process a sentence to word tokens
def process_sentence(sentence: str) -> list:
    """Creates a list of processed word tokens from a regular sentence.
   Args:
       sentence: sentence in string format.

   Returns:
       word_tokens: list containing all the tokens found in the sentences.
   """
    # set the stop words dictionary
    stop_words = set(stopwords.words('english'))

    # setting the lemmatizer dictionary
    lemmatizer = WordNetLemmatizer()

    # set all the words in the sentence to lowercase
    sentence = sentence.lower()

    # tokenize the words
    word_tokens = word_tokenize(sentence)

    # remove the stop words
    word_tokens = [w for w in word_tokens if w not in stop_words]

    # use lemmatization (reduce all word to their root form) to reduce the amount of words
    word_tokens = [lemmatizer.lemmatize(w) for w in word_tokens]

    # remove all the punctuation from sentence
    word_tokens = [w for w in word_tokens if w not in string.punctuation]

    # return the processed word tokens
    return word_tokens


def calculate_tf_idf(word: str, sentence: list, document: list) -> float:
    """Calculates the term frequency-inverse frequency document of a word into a sentence from a document corpus.

    Args:
        word: string, must be lowercase
        sentence: list of strings, all should be lowercase.
        document: list of sentences from a document corpus

    Returns:
        float number between 0 and 1, representing the product between tf (term frequency) and idf
        (inverse document frequency)
    """
    # the term frequency of the word (number of word appearances in the sentence/ total number of words in the sentence)
    tf = sentence.count(word) / len(sentence)

    # the inverse frequency document of the word ( log(total number of sentences in the documents/ number of sentences
    # where the word appears))
    n_sentences = 0
    for k in range(len(document)):
        if word in document[k]:
            n_sentences += 1
    # use 1 in the log to avoid division by 0
    idf = np.log(len(document) / n_sentences)

    # return the tf-idf of the word
    return tf*idf


# the type of data to be read (train or validation)
data_type = 'train'

# save the data in json format
with open(data_type + '_data.json', 'r') as json_file:
    data = json.load(json_file)
    print("The data has been loaded from", data_type + '_data.json')


# the text data
text_data = data['text_data']
print("The data contains a document with", len(text_data), "labelled examples.")

# process the data into word tokens (that can then be vectorized)
word_tokens_data = [process_sentence(text) for text in text_data]

# using TF-IDF encoding
encoded_tokens = []

# max length of an array (used later to set all the arrays to the same size)
max_length = 0

# going through each sentence
print("Encoding the data using tf-idf. This might take a while.")
for i in range(len(word_tokens_data)):
    # encoded tokens from one example
    encoded_example = [calculate_tf_idf(w, word_tokens_data[i], word_tokens_data) for w in word_tokens_data[i]]
    encoded_tokens.append(encoded_example)
    if max_length < len(encoded_example):
        max_length = len(encoded_example)

# the numpy array of encoded tokens
encoded_data = np.zeros((len(encoded_tokens), max_length))
print("Padding the data and transforming to numpy arrays. This might take a while.")
for i in range(len(encoded_tokens)):
    for j in range(len(encoded_tokens[i])):
        encoded_data[i][j] = encoded_tokens[i][j]

with open(data_type + '_encoded_data', 'wb') as handle:
    pickle.dump(encoded_data, handle)
    print("The data has been saved in", data_type + 'encoded_data')
    print("The shape of the data is:", encoded_data.shape)
