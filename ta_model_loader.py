# Custom model loader for textattack
import torch
import os
from textattack.models.wrappers import PyTorchModelWrapper
from project.utils.model_factory import construct_model_from_config

# Remember to set the PYTHONPATH environment variable to the root of the project
from project.utils import tokenizer
# Remember to set the TA_VICTIM_MODEL_PATH environment variable
assert os.environ.get(
    "TA_VICTIM_MODEL_PATH") is not None, "Please set the TA_VICTIM_MODEL_PATH environment variable"

model_path = os.environ.get("TA_VICTIM_MODEL_PATH")
output_dir = model_path[:model_path.rfind("/")]
config_file = f"{output_dir}/config.py"
print(f"Loading model from {model_path}")

my_model, Config, vocab, device = construct_model_from_config(config_file) 
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
