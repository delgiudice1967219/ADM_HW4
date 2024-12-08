import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import numpy as np


def extract_year(df, original_column='title', new_column='year'):
    """
    Extracts the year from the title column and creates a new column for the year.
    Removes the year from the title column.

    Inputs:
        df: DataFrame.
        original_column: Name of the column containing the year to extract.
        new_column: Name of the new column for the extracted years.

    Outputs:
        df: DataFrame with the new year column and cleaned title column.
    """
    # Regex pattern to recognize the year in format "(YYYY)"
    year_pattern = r'\((\d{4})\)'

    # Extract the year and remove it from the title
    df[new_column] = df[original_column].str.extract(year_pattern).astype(float)
    df[original_column] = df[original_column].str.replace(year_pattern, '', regex=True).str.strip()
    return df


def preprocess_text(doc):
    """
    Function that preprocesses a document by:
        Tokenizing, removing stopwords, punctuation, and stemming the tokens.

    Inputs:
        doc: document to preprocess

    Outputs:
        tokens: list of cleaned tokens
    """

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
    """
    Compute the embeddings mean vector for all the tokens in each cell.

    Inputs:
        model: Word2Vec model that contains numeric vectors for words (model.wv).
        tokens: List of tokenized and cleaned text
    Outputs:
        A numeric vector (embedding) that represents the mean embedding of the tokens in the input text.
        If no embeddings are found, a zero vector is returned.
    """
    # List that contains the embeddings of the tokens found in the model's vocabulary
    embeddings = []

    # Iterate through each token in each cell, considering that some cells has more than one token
    for token in tokens:
        # Check if the token exists in the model's vocabulary (i.e., it has an embedding)
        if token in model.wv:
            # Add the embedding of the token to the list of embeddings
            embeddings.append(model.wv[token])

    # If there are any valid embeddings(list is not empty)
    if embeddings:
        # Compute the mean of the embeddings to represent the text
        return np.mean(embeddings, axis=0)
    else:
        # If the embedding is not computed return a zero array of the same size as the model's embeddings
        return np.zeros(model.vector_size)