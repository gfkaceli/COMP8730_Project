import pytorch_lightning as pl
from GlossingModel import GlossingPipeline
from data import GlossingDataModule
from metrics import compute_morpheme_level_gloss_accuracy

# Override the word-level accuracy function
def compute_word_level_gloss_accuracy(predictions: list, targets: list) -> float:
    """
    Computes word-level glossing accuracy over a set of predictions.
    A predicted gloss is considered correct based on the percentage of matching tokens,
    ignoring <unk> tokens in the target.
    """
    if len(targets) == 0:
        return 1.0
    
    total_matching_tokens = 0
    total_tokens = 0
    
    for pred, target in zip(predictions, targets):
        # Split into tokens
        pred_tokens = pred.strip().split()
        target_tokens = target.strip().split()
        
        # Ignore <unk> tokens in the target
        target_tokens = [t for t in target_tokens if t != "<unk>"]
        
        # Truncate or pad predicted tokens to match the length of the target tokens
        if len(pred_tokens) < len(target_tokens):
            pred_tokens += ["<pad>"] * (len(target_tokens) - len(pred_tokens))
        elif len(pred_tokens) > len(target_tokens):
            pred_tokens = pred_tokens[:len(target_tokens)]
        
        # Count matching tokens
        matching_tokens = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
        total_matching_tokens += matching_tokens
        total_tokens += len(target_tokens)
    
    # Return the percentage of matching tokens
    return total_matching_tokens / total_tokens if total_tokens > 0 else 1.0

if __name__ == '__main__':
    pl.seed_everything(42, workers=True)

    language_code_mapping = {
        "Arapaho": "arp",
        "Gitksan": "git",
        "Lezgi": "lez",
        "Natugu": "ntu",
        "Nyangbo": "nyb",
        "Tsez": "ddo",
        "Uspanteko": "usp",
    }

    # Define file paths for training, validation, and test data.
    train_file = "data/Lezgi/lez-train-track1-uncovered"
    val_file = "data/Lezgi/lez-dev-track1-uncovered"
    test_file = "data/Lezgi/lez-test-track1-uncovered"

    # Create the DataModule instance.
    dm = GlossingDataModule(train_file=train_file, val_file=val_file, test_file=test_file, batch_size=7)
    dm.setup(stage="fit")
    dm.setup(stage="test")

    # Retrieve vocabulary sizes from the DataModule.
    char_vocab_size = dm.source_alphabet_size   # Source vocabulary size (for characters)
    gloss_vocab_size = dm.target_alphabet_size    # Gloss vocabulary size (for gloss tokens)
    trans_vocab_size = dm.trans_alphabet_size      # Translation vocabulary size

    # Define hyperparameters.
    embed_dim = 128
    num_heads = 16
    ff_dim = 512
    num_layers = 2
    dropout = 0.1
    use_gumbel = True
    learning_rate = 0.001
    use_relative = True
    max_relative_position = 16
    # Assume the gloss tokenizer uses "<pad>" as the padding token.
    gloss_pad_idx = dm.target_tokenizer["<pad>"]

    # Instantiate the integrated glossing model.
    model = GlossingPipeline(
        char_vocab_size=char_vocab_size,
        gloss_vocab_size=gloss_vocab_size,
        trans_vocab_size=trans_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_gumbel=use_gumbel,
        learning_rate=learning_rate,
        gloss_pad_idx=gloss_pad_idx,
        use_relative=use_relative,
        max_relative_position=max_relative_position,
    )

    # Configure the PyTorch Lightning Trainer.
    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="auto",
        log_every_n_steps=5,
        deterministic=True
    )

    # Train the model.
    trainer.fit(model, dm)

    # Save the trained model checkpoint.
    checkpoint_path = "models/glossing_model_gitksan.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    # Get predictions and true glosses.
    predictions = trainer.predict(model, dataloaders=dm.test_dataloader())

    # Create an inverse mapping for the gloss vocabulary.
    inv_gloss_vocab = {idx: token for token, idx in dm.train_dataset.gloss_vocab.items()}

    # Extract true glosses from the test dataset.
    true_glosses = []
    for batch in dm.test_dataloader():
        _, _, tgt_batch, _ = batch
        for tgt in tgt_batch:
            gloss_tokens = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in tgt if idx.item() != gloss_pad_idx]
            if "</s>" in gloss_tokens:
                gloss_tokens = gloss_tokens[:gloss_tokens.index("</s>")]
            true_gloss = " ".join(gloss_tokens)
            true_glosses.append(true_gloss)

    # Process and print predictions alongside true glosses.
    predicted_glosses = []
    sample_index = 0  # Global sample index across all batches
    print("\nPredictions and True Glosses:")
    for batch in predictions:
        for pred in batch:
            tokens = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in pred if idx.item() != gloss_pad_idx]
            if "</s>" in tokens:
                tokens = tokens[:tokens.index("</s>")]
            predicted_gloss = " ".join(tokens)
            predicted_glosses.append(predicted_gloss)
            # Print predicted gloss and true gloss side by side.
            print(f"Sample {sample_index + 1}:")
            print(f"  Predicted Gloss: {predicted_gloss}")
            print(f"  True Gloss:     {true_glosses[sample_index]}")
            print()  # Add a blank line for readability.
            sample_index += 1  # Increment the global sample index

    # Calculate and print word-level and morpheme-level gloss accuracy.
    word_level_accuracy = compute_word_level_gloss_accuracy(predicted_glosses, true_glosses)
    morpheme_level_accuracy = compute_morpheme_level_gloss_accuracy(predicted_glosses, true_glosses)

    print("\nWord-Level Gloss Accuracy:", word_level_accuracy)
    print("Morpheme-Level Gloss Accuracy:", morpheme_level_accuracy)