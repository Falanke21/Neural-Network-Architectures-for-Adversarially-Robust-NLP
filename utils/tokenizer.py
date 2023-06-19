import nltk
import string
import torchtext
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')  # Uncomment this line if you haven't downloaded nltk punkt

def tokenize(text: str, remove_stopwords: bool = False) -> list:
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
    tokens = nltk.word_tokenize(nopunc)
    if remove_stopwords:
        return [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    else:
        return [lemmatizer.lemmatize(word) for word in tokens]

class MyTokenizer():
    """
    Wrapper for textattack tokenizer
    """

    def __init__(self, vocab, seq_length, remove_stopwords: bool = False):
        """
        :param vocab: torchtext.vocab.Vocab object, a mapping from tokens to indices,
        or GloVe object from torchtext
        :param remove_stopwords: whether to remove stopwords using nltk
        """
        self.vocab = vocab
        if vocab and not isinstance(self.vocab, torchtext.vocab.Vocab) and not isinstance(
                self.vocab, torchtext.vocab.GloVe):
            raise ValueError(
                "Vocab must be either torchtext.vocab.Vocab or torchtext.vocab.GloVe")
        self.seq_length = seq_length
        self.remove_stopwords = remove_stopwords
        self.pad_token_id = 0  # default pad token id (Textattack usage only)

    def tokenize(self, text: str) -> list:
        '''
        Takes in a string of text, then performs the following:
        1. Remove all punctuation
        2. Remove all stopwords if remove_stopwords is True
        3. Lemmatize each word
        4. Return the cleaned text as a list of strings
        '''
        tokenize(text, self.remove_stopwords)

    def tokens_to_ids(self, token_list: list) -> list:
        """
        Return a list of ids for each token in the list of string tokens.
        """
        # Different vocab object has different ways to get the stoi mapping
        if isinstance(self.vocab, torchtext.vocab.Vocab):
            stoi = self.vocab.get_stoi()
        elif isinstance(self.vocab, torchtext.vocab.GloVe):
            stoi = self.vocab.stoi
        indices = self.seq_length * [0]  # initialize as 0s
        for i, token in enumerate(token_list):
            if i >= self.seq_length:
                # Reached the maximum sequence length
                break
            if token in stoi:
                indices[i] = stoi[token]
            else:
                # Unknown token
                indices[i] = stoi['<unk>'] if '<unk>' in stoi else 0
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

    def convert_id_to_word(self, id: int) -> str:
        """
        Convert an id to a word. (Textattack usage only)
        """
        if isinstance(self.vocab, torchtext.vocab.GloVe):
            return self.vocab.itos[id]
        elif isinstance(self.vocab, torchtext.vocab.Vocab):
            return self.vocab.get_itos()[id]
        else:
            raise ValueError(
                f"vocab must have either itos or get_itos() method, \
                    but got {type(self.vocab)} instead.")

    def convert_ids_to_tokens(self, ids: list) -> list:
        """
        Convert a list of ids to a list of words. (Textattack usage only)
        """
        return [self.convert_id_to_word(id) for id in ids]
