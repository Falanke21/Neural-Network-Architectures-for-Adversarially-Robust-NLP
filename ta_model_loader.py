# Custom model loader for textattack
import importlib
import pickle
import torch
from textattack.models.wrappers import PyTorchModelWrapper

# Remember to set the PYTHONPATH environment variable to the root of the project
from project.utils import tokenizer

# Choose the model type
MODEL_TYPE = "transformer"
# MODEL_TYPE = "transformer"
model_path = f"models/{MODEL_TYPE}_custom_embedding.pt"
config_name = f"{MODEL_TYPE}_default"

# Load the model
if MODEL_TYPE == "lstm":
    from project.lstm.my_lstm import MyLSTM
    Config = importlib.import_module('project.config.' + config_name).LSTMConfig
elif MODEL_TYPE == "transformer":
    from project.transformer.my_transformer import MyTransformer
    Config = importlib.import_module('project.config.' + config_name).TransformerConfig
print(f"Using config: {config_name}")


vocab_path = "data/vocab200k.pkl"
with open(vocab_path, 'rb') as f:
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


my_model = ModelWithSigmoid(my_model)
# print(f"Model structure: {my_model}")
# Load the tokenizer
tokenizer = tokenizer.MyTokenizer(
    vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)
# Wrap the model with Textattack's wrapper
model = PyTorchModelWrapper(my_model, tokenizer)
