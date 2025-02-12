#importing PyTorch and other libraries
import torch
from torch.utils.data import Dataset, DataLoader
#Had to modify T5tokenizer library since it used sentencepiece which is not compatible with Python 3.12
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, Trainer, TrainingArguments
import pandas as pd #to read csv files

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
def load_dataset(file_path):
    df = pd.read_csv(r'D:\University of Windsor\Spring 2025\COMP 8730 - NLP\Dataset\Dataset1.csv')  # Ensure the CSV has 'input' and 'target' columns
    return list(zip(df['orthographic'].tolist(), df['translation'].tolist(), df['gloss'].tolist()))

def create_prompt(row):
    return f"gloss the sentence:   {row['orthographic']}  {row['translation']} " 

# Tokenizer
#switched over from t5-small to t5-base
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=True)

# Dataset Class
class GlossingDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=128): #changed from max_len=50
        self.inputs = [
            #tokenizer.encode("orthographic: " + ortho + " translation: " + trans, padding="max_length", max_length=max_len, truncation=True)
            tokenizer.encode(f"gloss the sentence: {trans} {ortho} gloss {gloss}", padding="max_length", max_length=max_len, truncation=True)
            for ortho, trans, gloss in pairs
        ]
        self.targets = [
            tokenizer.encode(gloss, padding="max_length", max_length=max_len, truncation=True)
            for _, _, gloss in pairs
        ]
        self.max_len = max_len
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.targets[idx], dtype=torch.long)
        }
    

data = load_dataset("Dataset1.csv")  

# Create dataset and dataloader
dataset = GlossingDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Loading T5 Model
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

# Training loop
EPOCHS = 50
for epoch in range(EPOCHS):
    loss = train_epoch(model, dataloader, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Evaluation
model.eval()
def gloss_text(orthographic, translation):
    #input_text = "orthographic: " + orthographic + " translation: " + translation
    input_text = f"gloss: {translation} {orthographic}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    #output_ids = model.generate(input_ids)
    output_ids = model.generate(input_ids, max_length=128, num_beams=5, num_return_sequences=1, early_stopping=True)
    glossed_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #glossed_text = model.predict(output_ids)[0]

    glossed_text = glossed_text.replace(" .", "").replace(" ,", "").strip()

    #orthographic_line = f"\\t {orthographic}"

    #gloss_line = f"\\g {glossed_text}"

    #translation_line = f"\\l {translation}" 
    
    #return orthographic_line, gloss_line, translation_line

    return f"\\t {orthographic}", f"\\g {glossed_text}", f"\\l {translation}"


# Test Example
orthographic, gloss, translation = gloss_text("Neneenin beehiniisonoonibeihin", "You the father of all")
print(orthographic)
print(gloss)
print(translation)
#glossed = gloss_text(input_sentence, translated_sentence)
#print("Orthographic:", input_sentence)
#print("Translation:", translated_sentence)
#print("Glossed:", glossed)