import torch
from backbone import BiaffineDependencyParser
from dataset import Vocab
from datasets import load_dataset
import os

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Load dataset to create vocab
ds = load_dataset("universal_dependencies", "en_ewt", cache_dir='/media/amax/xiao_20T1/code/jly/DP/dataset')
vocab = Vocab(ds)

# Initialize model
parser = BiaffineDependencyParser(
    vocab_size=len(vocab.word_to_id), 
    upos_size=len(vocab.upos_to_id),
    deprel_size=len(vocab.deprel_to_id),
    embed_dim=384,             
    hidden_dim=1536,            
    lstm_layers=4,             
    ffnn_dim=384,              
    num_heads=8
)

# Calculate total parameters
total_params = sum(p.numel() for p in parser.parameters())
print(f"Total trainable parameters: {total_params}")

# Estimate model size (assuming float32, 4 bytes per param)
estimated_size_mb = (total_params * 4) / (1024 * 1024)
print(f"Estimated checkpoint size: {estimated_size_mb:.2f} MB")

# Save a dummy checkpoint to check actual size
torch.save(parser.state_dict(), 'dummy_checkpoint.pth')
actual_size = os.path.getsize('dummy_checkpoint.pth') / (1024 * 1024)
print(f"Actual dummy checkpoint size: {actual_size:.2f} MB")
os.remove('dummy_checkpoint.pth')