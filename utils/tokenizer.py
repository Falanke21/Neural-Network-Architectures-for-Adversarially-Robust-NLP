import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def tokenize(text: str) -> list:
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Lemmatize each word
    4. Return the cleaned text as a list of words
    '''
    text = text.lower()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in nopunc.split() if word not in stopwords.words('english')]
