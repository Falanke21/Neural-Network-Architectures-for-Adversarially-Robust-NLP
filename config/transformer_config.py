class TransformerConfig:
    USE_GPU = True
    NUM_EPOCHS = 5
    TRAIN_SEQ_LENGTH = 150
    TEST_SEQ_LENGTH = 450
    BATCH_SIZE = 200
    LEARNING_RATE = 0.001
    UPSAMPLE_NEGATIVE = True
    UPSAMPLE_RATIO = 2  # 1 means no upsampling

    D_MODEL = 300
    FFN_HIDDEN = 1024
    N_HEAD = 5
    DROPOUT = 0.1
