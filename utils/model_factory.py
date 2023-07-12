# Get model from config file
import importlib
import pickle
import torch
import os


def construct_model_from_config(config_path: str):
    assert os.environ["MODEL_CHOICE"] in [
        'lstm', 'transformer'], "Env var MODEL_CHOICE must be either 'lstm' or 'transformer'"
    # load Config object from config file
    # check config file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
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
    if Config.USE_GLOVE:
        import torchtext
        glove = torchtext.vocab.GloVe(
            name='6B', dim=Config.GLOVE_EMBEDDING_SIZE,
            cache=Config.GLOVE_CACHE_DIR)
        vocab = glove
    else:
        with open(Config.CUSTOM_VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)

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
