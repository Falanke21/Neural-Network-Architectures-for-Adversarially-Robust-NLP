# Custom model loader for textattack
import importlib
import pickle
import torch
import os
from textattack.models.wrappers import PyTorchModelWrapper

# Remember to set the PYTHONPATH environment variable to the root of the project
from project.utils import tokenizer
# Remember to set the TA_VICTIM_MODEL_PATH environment variable
assert os.environ.get(
    "TA_VICTIM_MODEL_PATH") is not None, "Please set the TA_VICTIM_MODEL_PATH environment variable"

# Choose the model type
# MODEL_TYPE = "lstm"
MODEL_TYPE = "transformer"
EMBEDDING = "custom"
# EMBEDDING = "glove"
model_path = os.environ.get("TA_VICTIM_MODEL_PATH")
output_dir = model_path[:model_path.rfind("/")]
config_file = f"{output_dir}/config.py"
print(f"Loading model from {model_path}")

# import configs
spec = importlib.util.spec_from_file_location("Config", config_file)
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
if MODEL_TYPE == 'lstm':
    Config = config_module.LSTMConfig
    from project.lstm.my_lstm import MyLSTM
elif MODEL_TYPE == 'transformer':
    Config = config_module.TransformerConfig
    from project.transformer.my_transformer import MyTransformer
print(f"Using config {Config.__name__} from {config_file}")

# Load the model
if EMBEDDING == "custom":
    vocab_path = "data/vocab300k.pkl"
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
elif EMBEDDING == "glove":
    import torchtext
    glove = torchtext.vocab.GloVe(
        name='6B', dim=Config.GLOVE_EMBEDDING_SIZE,
        cache=Config.GLOVE_CACHE_DIR)
    vocab = glove

device = torch.device(
    'cuda' if Config.USE_GPU and torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device index: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(
        f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3} GB')

if MODEL_TYPE == "lstm":
    my_model = model = MyLSTM(Config=Config, vocab_size=len(
        vocab), num_classes=1, device=device).to(device)
elif MODEL_TYPE == "transformer":
    my_model = MyTransformer(Config=Config, vocab_size=len(
        vocab), output_dim=1, device=device).to(device)
my_model.load_state_dict(torch.load(model_path))
my_model.eval()


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


my_model = ModelWithSigmoid(my_model)
# Load the tokenizer
tokenizer = tokenizer.MyTokenizer(
    vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)
# Wrap the model with Textattack's wrapper
model = PyTorchModelWrapper(my_model, tokenizer)
