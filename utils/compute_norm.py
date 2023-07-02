import csv
import importlib
import pickle
import torch

# Choose the model type
# MODEL_TYPE = "lstm"
MODEL_TYPE = "transformer"
EMBEDDING = "custom"
# EMBEDDING = "glove"

output_dir = f"/homes/fh422/ic/project/vol_folder/model_zoo/data300k-with-3stars/transformer_custom_weight_decay1e3"
model_path = f"{output_dir}/transformer_model_epoch42.pt"

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
    vocab_path = "/homes/fh422/ic/project/data/vocab300k.pkl"
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


def compute_norm(my_model: torch.nn.Module):
    # Compute the norm
    sum_of_norm = 0
    sum_of_norm_no_embedding = 0
    for name, param in my_model.named_parameters():
        if param.requires_grad:
            norm = torch.linalg.norm(param).item()
            sum_of_norm += norm
            if name != "embedding.weight":
                # we separately compute the norm of embedding
                sum_of_norm_no_embedding += norm

    print(f"Sum of norm: {sum_of_norm:.2f}")
    print(f"Sum of norm without embedding: {sum_of_norm_no_embedding:.2f}")
    return sum_of_norm, sum_of_norm_no_embedding


compute_norm(my_model)


# # Now, we compute the same thing but for every epochs
# checkpoint_dir = f"{output_dir}/checkpoints"
# for e in range(1, Config.NUM_EPOCHS + 1):
#     model_path = "{}/transformer_model_epoch{}.pt".format(checkpoint_dir, e)
#     my_model.load_state_dict(torch.load(model_path))
#     my_model.eval()

#     print(f"Epoch {e}")
#     sum_of_norm, sum_of_norm_no_embedding = compute_norm(my_model)

#     # Write to csv
#     with open(f"{output_dir}/norm.csv", 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow([e, sum_of_norm, sum_of_norm_no_embedding])
