class LSTMConfig:
    USE_GPU = True

    WORD_EMBEDDING = "custom"  # "custom", "glove" or "paragramcf"
    # Custom word embedding settings
    CUSTOM_VOCAB_PATH = 'data/vocab300k.pkl'
    # GloVe word embedding settings
    GLOVE_CACHE_DIR = '/vol/bitbucket/fh422/torchtext_cache'
    GLOVE_EMBEDDING_SIZE = 300
    # Paragramcf word embedding settings
    PARAGRAMCF_DIR = '/vol/bitbucket/fh422/paragramcf'

    NUM_EPOCHS = 50
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001

    USE_ADAMW = False
    BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP = True
    GRADIENT_CLIP_VALUE = 1
    UPSAMPLE_NEGATIVE = True
    UPSAMPLE_RATIO = 2  # 1 means no upsampling
    LABEL_SMOOTHING = False
    LABEL_SMOOTHING_EPSILON = 0.1  # 0 means no smoothing

    LSTM_HIDDEN_SIZE = 300
    LSTM_EMBEDDING_SIZE = 300
    LSTM_NUM_LAYERS = 1
    LSTM_DROUPOUT = 0
