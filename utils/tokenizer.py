import nltk
import string
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')  # Uncomment this line if you haven't downloaded nltk punkt


class MyTokenizer():
    """
    Wrapper for textattack tokenizer
    """

    def __init__(self, vocab, seq_length, remove_stopwords: bool = False):
        """
        :param vocab: torchtext.vocab.Vocab object, a mapping from tokens to indices
        :param remove_stopwords: whether to remove stopwords using nltk
        """
        self.vocab = vocab
        self.seq_length = seq_length
        self.remove_stopwords = remove_stopwords

    def tokenize(self, text: str) -> list:
        '''
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords if remove_stopwords is True
        3. Lemmatize each word
        4. Return the cleaned text as a list of strings
        '''
        text = text.lower()
        # Remove punctuation
        nopunc = [char for char in text if char not in string.punctuation]
        nopunc = ''.join(nopunc)

        lemmatizer = WordNetLemmatizer()
        tokens = nltk.word_tokenize(text)
        if self.remove_stopwords:
            return [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
        else:
            return [lemmatizer.lemmatize(word) for word in tokens]

    def tokens_to_ids(self, token_list: list) -> list:
        """
        Return a list of ids for each token in the list of string tokens.
        """
        indices = self.seq_length * [0]  # initialize as 0s
        for i, token in enumerate(token_list):
            if i >= self.seq_length:
                # Reached the maximum sequence length
                break
            if token in self.vocab:
                indices[i] = self.vocab[token]
            else:
                # Unknown token
                indices[i] = self.vocab['<unk>']
        return indices

    def __call__(self, text) -> list:
        """
        Return either a list of ids for each token in the text, 
        or a list of lists of ids because
        text might be a list of strings because of textattack
        """
        if isinstance(text, str):
            # If text is a string, we just tokenize it and return a list of ids(words)
            result = self.tokenize(text)
            return self.tokens_to_ids(result)
        elif isinstance(text, list):
            # If text is a list of strings, we tokenize each string and return a list of lists of ids
            result = []
            for t in text:
                result.append(self.tokens_to_ids(self.tokenize(t)))
            return result
        else:
            raise ValueError(
                f"Input text must be either a string or a list of strings, but got {type(text)} instead.")
