import nltk
import string
import torchtext
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Uncomment this line if you haven't downloaded nltk punkt


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
        :param vocab: a mapping object/dict from tokens to indices,
        :param remove_stopwords: whether to remove stopwords using nltk
        """
        if vocab and not isinstance(vocab, torchtext.vocab.Vocab) and not isinstance(
                vocab, torchtext.vocab.GloVe) and not isinstance(vocab, dict):
            raise ValueError(
                "Vocab must be either torchtext.vocab.Vocab or torchtext.vocab.GloVe or dict")
        # Different vocab object has different ways to get the mappings
        if isinstance(vocab, torchtext.vocab.Vocab):
            self.word2id = vocab.get_stoi()
            self.id2word = vocab.get_itos()
        elif isinstance(vocab, torchtext.vocab.GloVe):
            self.word2id = vocab.stoi
            self.id2word = vocab.itos
        elif isinstance(vocab, dict):
            self.word2id = vocab
            self.id2word = {}
            for word, index in vocab.items():
                self.id2word[index] = word

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
        return tokenize(text, self.remove_stopwords)

    def tokens_to_ids(self, token_list: list) -> list:
        """
        Return a list of ids for each token in the list of string tokens.
        """
        indices = self.seq_length * [0]  # initialize as 0s
        for i, token in enumerate(token_list):
            if i >= self.seq_length:
                # Reached the maximum sequence length
                break
            if token in self.word2id:
                indices[i] = self.word2id[token]
            else:
                # Unknown token
                indices[i] = self.word2id['<unk>'] if '<unk>' in self.word2id else 0
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
        if id == self.pad_token_id:
            return '<pad>'
        return self.id2word[id]

    def convert_ids_to_tokens(self, ids: list) -> list:
        """
        Convert a list of ids to a list of words. (Textattack usage only)
        """
        result = []
        for id in ids:
            if id != self.pad_token_id:
                result.append(self.convert_id_to_word(id))
        return result
