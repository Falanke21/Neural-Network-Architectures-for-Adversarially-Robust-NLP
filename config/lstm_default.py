class LSTMConfig:
    USE_GPU = True
    USE_GLOVE = False
    GLOVE_CACHE_DIR = '/vol/bitbucket/fh422/torchtext_cache'
    GLOVE_EMBEDDING_SIZE = 300
    NUM_EPOCHS = 50
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001

    BETAS = (0.9, 0.999)
    ADAM_EPSILON = 1e-8
    WEIGHT_DECAY = 1e-5
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
