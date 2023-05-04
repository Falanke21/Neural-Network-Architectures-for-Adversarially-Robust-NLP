class TransformerConfig:
    USE_GPU = True
    NUM_EPOCHS = 5
    TRAIN_SEQ_LENGTH = 150
    TEST_SEQ_LENGTH = 450
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001

    BETAS = (0.9, 0.98)  # same as original paper
    ADAM_EPSILON = 1e-9  # same as original paper
    WEIGHT_DECAY = 0  # 0 means no weight decay
    GRADIENT_CLIP = True
    GRADIENT_CLIP_VALUE = 1
    UPSAMPLE_NEGATIVE = True
    UPSAMPLE_RATIO = 2  # 1 means no upsampling
    LABEL_SMOOTHING = False  # label smoothing lead to worse results
    LABEL_SMOOTHING_EPSILON = 0.1  # 0 means no smoothing

    D_MODEL = 300
    FFN_HIDDEN = 1024
    N_HEAD = 5
    DROPOUT = 0.1
