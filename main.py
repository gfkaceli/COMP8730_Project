import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pytorch_lightning as pl
from GlossingModel import GlossingPipeline

#########################################
# 1. Custom Dataset for CSV Data
#########################################
class GlossingDataset(Dataset):
    def __init__(self, csv_file, max_src_len=50, max_tgt_len=50, max_trans_len=50):
        self.data = pd.read_csv(csv_file).dropna().reset_index(drop=True)  # Remove empty rows
        
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len
        
        # Build vocabularies dynamically
        self.src_vocab = self.build_vocab(self.data["Language"], char_level=True)
        self.gloss_vocab = self.build_vocab(self.data["Gloss"], char_level=False)
        self.trans_vocab = self.build_vocab(self.data["Translation"], char_level=False)

    def build_vocab(self, data, char_level=False):
        counter = Counter()
        for item in data.dropna():
            tokens = list(item) if char_level else item.split()
            counter.update(tokens)
        vocab = {tok: i for i, tok in enumerate(counter.keys(), start=2)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        return vocab

    def text_to_tensor(self, text, vocab, char_level=False):
        tokens = list(text) if char_level else text.split()
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_text(self, tensor, vocab):
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        return " ".join([inv_vocab.get(idx, "<unk>") for idx in tensor.tolist()])


    def __len__(self):
        return len(self.data)

    def tokenize(self, text, vocab, max_len, char_level=False):
        tokens = list(text) if char_level else text.split()
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        return indices

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        src_indices = self.tokenize(str(row["Language"]), self.src_vocab, self.max_src_len, char_level=True)
        gloss_indices = self.tokenize(str(row["Gloss"]), self.gloss_vocab, self.max_tgt_len)
        trans_indices = self.tokenize(str(row["Translation"]), self.trans_vocab, self.max_trans_len)

        return (torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(gloss_indices, dtype=torch.long),
                torch.tensor(trans_indices, dtype=torch.long))

def collate_fn(batch):
    src_batch, tgt_batch, trans_batch = zip(*batch)
    return (torch.stack(src_batch, dim=0),
            torch.stack(tgt_batch, dim=0),
            torch.stack(trans_batch, dim=0))

#########################################
# 2. Training Script using PyTorch Lightning
#########################################
if __name__ == '__main__':
    pl.seed_everything(42)
    
    # Load dataset
    dataset = GlossingDataset("data/Dummy_Dataset.csv", max_src_len=50, max_tgt_len=20, max_trans_len=10)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    # Model parameters
    char_vocab_size = len(dataset.src_vocab)
    gloss_vocab_size = len(dataset.gloss_vocab)
    trans_vocab_size = len(dataset.trans_vocab)
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_layers = 4  # Reduced to avoid overfitting
    dropout = 0.3
    use_gumbel = True
    learning_rate = 0.001
    gloss_pad_idx = dataset.gloss_vocab["<pad>"]

    # Initialize model
    model = GlossingPipeline(char_vocab_size, gloss_vocab_size, trans_vocab_size,
                             embed_dim, num_heads, ff_dim, num_layers, dropout, use_gumbel,
                             learning_rate, gloss_pad_idx)
    
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", log_every_n_steps=5)
    trainer.fit(model, dataloader)
    
    # Saving the trained model here
    trainer.save_checkpoint("glossing_model.ckpt")

#########################################
# 3. Inference Function
#########################################
def predict_gloss(model, dataset, source_text, translation_text, max_len=20):
    # Convert input text to tensors
    src_tensor = dataset.text_to_tensor(source_text, dataset.src_vocab, char_level=True).unsqueeze(0)  # (1, seq_len)
    trans_tensor = dataset.text_to_tensor(translation_text, dataset.trans_vocab, char_level=False).unsqueeze(0)  # (1, seq_len)

    # Convert source tensor to one-hot encoding (required for encoder)
    src_tensor = F.one_hot(src_tensor, num_classes=len(dataset.src_vocab)).float()

    # Start with an empty output sequence
    tgt_tensor = torch.zeros((1, 1), dtype=torch.long).to(model.device)  # (1, 1) placeholder

    generated_tokens = []

    for _ in range(max_len):  # Generate up to max_len tokens
        gloss_logits, _, _, _ = model(src_tensor, torch.tensor([src_tensor.shape[1]]), tgt_tensor, trans_tensor)
        
        # Getting the next token
        next_token = torch.argmax(gloss_logits[:, -1, :], dim=-1).item()
        
        if next_token == dataset.gloss_vocab["<pad>"]:  # Stop if padding token is predicted
            break

        generated_tokens.append(next_token)

        # Append new token to tgt_tensor
        tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]], dtype=torch.long).to(model.device)], dim=1)

    # Convert generated indices to gloss text
    gloss_text = dataset.tensor_to_text(torch.tensor(generated_tokens), dataset.gloss_vocab)
    return gloss_text


#########################################
# 4. Load Model and Run Inference
#########################################

dataset = GlossingDataset("data/Dummy_Dataset.csv")


trained_model = GlossingPipeline.load_from_checkpoint("glossing_model.ckpt")

# Test Run
print("The predicted gloss for language 'inopi-a' with translation 'a wine shortage' is:")
print(predict_gloss(trained_model, dataset, "inopi-a", "a wine shortage"))
