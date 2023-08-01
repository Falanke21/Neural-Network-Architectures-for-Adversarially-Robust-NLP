# Custom model loader for textattack
import torch
import os
from textattack.models.wrappers import PyTorchModelWrapper
from project.utils.model_factory import construct_model_from_config, ModelWithSigmoid

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


my_model = ModelWithSigmoid(my_model)
# Load the tokenizer
model_tokenizer = tokenizer.MyTokenizer(
    vocab, Config.MAX_SEQ_LENGTH, remove_stopwords=False)
# Wrap the model with Textattack's wrapper
model = PyTorchModelWrapper(my_model, model_tokenizer)
