import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import numpy as np


def extract_year(df, title_column='title', year_column='year'):
    '''
    Extracts the year from the title column and creates a new column for the year.
    Removes the year from the title column.

    Inputs:
    - df: DataFrame.
    - title_column: Name of the column containing the titles.
    - year_column: Name of the new column for the extracted years.

    Outputs:
    - df: DataFrame with the new year column and cleaned title column.
    '''
    # Regex pattern to recognize the year in format "(YYYY)"
    year_pattern = r'\((\d{4})\)'

    # Extract the year and remove it from the title
    df[year_column] = df[title_column].str.extract(year_pattern).astype(float)
    df[title_column] = df[title_column].str.replace(year_pattern, '', regex=True).str.strip()
    return df


def preprocess_text(doc):
    '''
    Function that preprocesses a document by:
    - Tokenizing, removing stopwords, punctuation, and stemming the tokens.

    Input:
    doc: document to preprocess

    Output:
    tokens: list of cleaned tokens
    '''

    # Check if the input is a string, if not return an empty list
    if not isinstance(doc, str):
        return []

    # Tokenize the document
    tokens = word_tokenize(doc)

    # Turn all words to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stops = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stops]

    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def get_embedding(model, tokens):
    # Compute the embeddings mean for each token in each cell
    embeddings = []
    # Consider that some cell has more than one word so in the tokenization process this will become a list of words
    for token in tokens:
        # Check if the token is beign computed by the model
        if token in model.wv:
            # Join the embedding for each world
            embeddings.append(model.wv[token])
    if embeddings:
        # Compute the mean of the embedding vector
        return np.mean(embeddings, axis=0)
    else:
        # If the embedding is not computed return a full of zero array
        return np.zeros(model.vector_size)