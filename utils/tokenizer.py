import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')


def tokenize(text: str, remove_stopwords: bool = False) -> list:
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords if remove_stopwords is True
    3. Lemmatize each word
    4. Return the cleaned text as a list of words
    '''
    text = text.lower()
    # Remove punctuation
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    if remove_stopwords:
        return [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    else:
        return [lemmatizer.lemmatize(word) for word in tokens]
