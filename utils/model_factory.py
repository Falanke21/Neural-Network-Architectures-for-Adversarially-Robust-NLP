# Get model from config file
import importlib
import numpy as np
import pickle
import torch
import os


def validate_config_path(config_path: str):
    if not os.path.isfile(config_path):
        # two cases:
        # 1. config file does not exist
        # 2. current working directory is checkpoints and config file is in parent directory
        # now we handle case 2
        if os.path.basename(os.path.dirname(config_path)) != 'checkpoints':
            raise FileNotFoundError(f"Config file {config_path} not found")
        else:
            print(f"Config file {config_path} not found, using config from parent directory")
            config_path = os.path.join(os.path.dirname(os.path.dirname(config_path)), 'config.py')
            print(f"New config file path: {config_path}")
    return config_path


def construct_model_from_config(config_path: str):
    assert os.environ["MODEL_CHOICE"] in [
        'lstm', 'transformer'], "Env var MODEL_CHOICE must be either 'lstm' or 'transformer'"
    # load Config object from config file
    config_path = validate_config_path(config_path)
    # import configs
    spec = importlib.util.spec_from_file_location("Config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if os.environ["MODEL_CHOICE"] == 'lstm':
        Config = config_module.LSTMConfig
        from project.lstm.my_lstm import MyLSTM
    elif os.environ["MODEL_CHOICE"] == 'transformer':
        Config = config_module.TransformerConfig
        # from transformer.my_transformer import MyTransformer
        from project.transformer.my_transformer import MyTransformer
    print(f"Using config {Config.__name__} from {config_path}")

    # load custom vocab or GloVe
    if Config.WORD_EMBEDDING == 'custom':
        with open(Config.CUSTOM_VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
    elif Config.WORD_EMBEDDING == 'glove':
        import torchtext
        glove = torchtext.vocab.GloVe(
            name='6B', dim=Config.GLOVE_EMBEDDING_SIZE,
            cache=Config.GLOVE_CACHE_DIR)
        vocab = glove
    elif Config.WORD_EMBEDDING == 'paragramcf':
        word_list_file = os.path.join(Config.PARAGRAMCF_DIR, 'wordlist.pickle')
        word2index = np.load(word_list_file, allow_pickle=True)
        vocab = word2index
    else:
        raise ValueError(
            "Config.WORD_EMBEDDING must be one of 'custom', 'glove' and 'paragramcf'")

    device = torch.device(
        'cuda' if Config.USE_GPU and torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(f'Device count: {torch.cuda.device_count()}')
        print(f'Current device index: {torch.cuda.current_device()}')
        print(f'Device name: {torch.cuda.get_device_name(0)}')
        print(
            f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')

    # define model
    if os.environ["MODEL_CHOICE"] == 'lstm':
        model = MyLSTM(Config=Config, vocab_size=len(
            vocab), num_classes=1, device=device).to(device)
    elif os.environ["MODEL_CHOICE"] == 'transformer':
        model = MyTransformer(Config=Config, vocab_size=len(
            vocab), output_dim=1, device=device).to(device)

    return model, Config, vocab, device


class ModelWithSigmoid(torch.nn.Module):
    """
    Our models only return logits, so we need to wrap it with a sigmoid layer.
    Also Textattack require the binary classification model to return the probability of 2 classes,
    we only have 1 class, so we need to add a dummy class with probability 1 - p.
    """

    def __init__(self, model):
        super(ModelWithSigmoid, self).__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)
        output = torch.sigmoid(logits)
        return torch.cat((1 - output, output), dim=1)

    def get_input_embeddings(self):
        """
        Return the input embedding layer
        """
        return self.model.embedding
