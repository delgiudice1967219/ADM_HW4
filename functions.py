import re
import nltk
nltk.download('punkt_tab')  #Used for tokenization
nltk.download('wordnet')  #Provides the lexical database needed for lemmatization.

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english')) 

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def create_shingles(string):
    '''
    The function takes a string as input and preprocesses it to create unigram shingles.
    These shingles are prepared tokens that are cleaned and normalized, making them suitable
    for further tasks such as minhashing.
    '''

    if not isinstance(string, str):
        return []
    string = string.lower()
    string = re.sub(r'\s+', ' ', string).strip()
    string = re.sub(r'-', ' ', string) 
    tokens = word_tokenize(string)
    shingles = [token for token in tokens if token.isalnum()]
    shingles = [token for token in shingles if token not in stop_words]

    return shingles
