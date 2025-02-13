import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
from EditDistance import minimum_edit_distance
# ============================================
# 1. Load and Prepare Your Data
# ============================================
# Assumes a CSV file "your_data.csv" with columns: "input_sentence", "output_translation", "gloss"
df = pd.read_csv("Dataset1.csv")


# Create the prompt from the input sentence and its translation.
def create_prompt(row):
    return f"Generate gloss for: input: {row['orthographic']} translation: {row['translation']}"


input_texts = df.apply(create_prompt, axis=1).tolist()
target_texts = df['gloss'].tolist()

# ============================================
# 2. Load the Tokenizer and Model
# ============================================
model_name = "t5-base"  # Change to your preferred model variant
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ============================================
# 3. Tokenize the Data
# ============================================
input_encodings = tokenizer(
    input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
)
target_encodings = tokenizer(
    target_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
)

# Replace pad_token_id in labels with -100 so that they are ignored in the loss computation
labels = target_encodings["input_ids"]
labels[labels == tokenizer.pad_token_id] = -100


# ============================================
# 4. Create a PyTorch Dataset
# ============================================
class GlossDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return self.labels.size(0)


dataset = GlossDataset(input_encodings, labels)

# ============================================
# 5. Set Up the Custom Training Loop
# ============================================
# Hyperparameters
batch_size = 4
num_epochs = 12
learning_rate = 0.001
torch.manual_seed(42)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Setup optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
model.train()  # set model to training mode
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0
    for batch in dataloader:
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        optimizer.zero_grad()

        # Forward pass: the model returns a dictionary that includes the loss when labels are provided.
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )
        loss = outputs.loss
        epoch_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Average Loss: {avg_loss:.4f}")


# ============================================
# 6. Define a Function to Generate a Gloss
# ============================================
def generate_gloss(input_sentence, output_translation, tokenizer, model, max_length=128):
    """
    Given an input sentence and its translation, generate a gloss.
    """
    # Create the prompt in the same format as the training data
    prompt = f"Generate gloss for: input: {input_sentence} translation: {output_translation}"

    # Tokenize the prompt and move it to the appropriate device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    # Generate output tokens
    generated_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=5,  # you can adjust beam search parameters as needed
        early_stopping=True
    )

    # Decode the token IDs to a human-readable string
    gloss = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return gloss


# ============================================
# 7. Test the Model on a Sample Input
# ============================================
if __name__ == "__main__":
    # Example usage
    sample_input_sentence = "Noh neihoowbeet3eiisin"
    sample_output_translation = "And I don't want to be in jail ."
    ground_truth_label = "and 1.NEG-want.to-be.in.jail"

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        generated_gloss = generate_gloss(sample_input_sentence, sample_output_translation, tokenizer, model)
        print(minimum_edit_distance(sample_input_sentence, generated_gloss))
    print("Generated gloss:", generated_gloss)