class TransformerConfig:
    USE_GPU = True
    WORD_EMBEDDING = "paragramcf"  # "custom", "glove" or "paragramcf"
    # Custom word embedding settings
    CUSTOM_VOCAB_PATH = 'data/vocab300k.pkl'
    # GloVe word embedding settings
    GLOVE_CACHE_DIR = '/vol/bitbucket/fh422/torchtext_cache'
    GLOVE_EMBEDDING_SIZE = 300
    # Paragramcf word embedding settings
    PARAGRAMCF_DIR = '/vol/bitbucket/fh422/paragramcf'
    NUM_EPOCHS = 50
    NUM_ADV_EPOCHS = 1  # Number of adversarial training epochs
    MAX_SEQ_LENGTH = 150
    BATCH_SIZE = 200
    LEARNING_RATE = 1e-4

    USE_ADAMW = False
    BETAS = (0.9, 0.98)  # same as original paper
    ADAM_EPSILON = 1e-9  # same as original paper
    WEIGHT_DECAY = 1e-7  # 0 means no weight decay
    GRADIENT_CLIP = True
    GRADIENT_CLIP_VALUE = 1
    UPSAMPLE_NEGATIVE = True
    UPSAMPLE_RATIO = 2  # 1 means no upsampling
    LABEL_SMOOTHING = False  # label smoothing lead to worse results
    LABEL_SMOOTHING_EPSILON = 0.1  # 0 means no smoothing

    NUM_LAYERS = 4
    D_MODEL = 300
    FFN_HIDDEN = 1024
    N_HEAD = 5
    DROPOUT = 0.2

    # 'dot_product' or 'additive' or 'paas' or 'paas-linear' or 
    # 'simal1' or 'simal2' or 'soft' or 'linformer' or 'cosformer'
    # or 'norm' or 'diag' or 'local' or 'experiment'
    # or 'transnormer' or 'robust' or 'reva' or 'revcos'
    ATTENTION_TYPE = 'dot_product'
    LINFORMER_K = 64
    DIAG_BLOCK_SIZE = 15
    NORM_ATTENTION_TYPE = 'layer-norm' # 'layer-norm' or 'srms'
    POSITIONAL_ENCODING = True  # Default is True
    FFN_TYPE = 'standard'  # 'standard' or 'glu'
    MH_TYPE = 'split'  # 'split' or 'parallel'

    # An extra regularization term for sum of ReLU outputs
    RELU_REGULARIZATION = False
    RELU_REGULARIZATION_LAMBDA = 1e-5
