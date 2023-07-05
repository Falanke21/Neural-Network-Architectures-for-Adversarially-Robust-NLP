class TransformerConfig:
    USE_GPU = True
    USE_GLOVE = False
    GLOVE_CACHE_DIR = '/vol/bitbucket/fh422/torchtext_cache'
    GLOVE_EMBEDDING_SIZE = 300
    NUM_EPOCHS = 50
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 200
    LEARNING_RATE = 1e-4

    BETAS = (0.9, 0.98)  # same as original paper
    ADAM_EPSILON = 1e-9  # same as original paper
    WEIGHT_DECAY = 1e-5  # 0 means no weight decay
    GRADIENT_CLIP = True
    GRADIENT_CLIP_VALUE = 1
    UPSAMPLE_NEGATIVE = True
    UPSAMPLE_RATIO = 2  # 1 means no upsampling
    LABEL_SMOOTHING = False  # label smoothing lead to worse results
    LABEL_SMOOTHING_EPSILON = 0.1  # 0 means no smoothing

    NUM_LAYERS = 1
    D_MODEL = 340
    FFN_HIDDEN = 1024
    N_HEAD = 5
    DROPOUT = 0.2

    ATTENTION_TYPE = 'dot_product'  # 'dot_product' or 'additive' or 'paas'
    POSITIONAL_ENCODING = True  # Default is True
