import pytorch_lightning as pl
from GlossingModel import GlossingPipeline
from data import GlossingDataModule

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
    train_file = "data/Gitksan/git-train-track1-uncovered"
    val_file = "data/Gitksan/git-dev-track1-uncovered"
    test_file = "data/Gitksan/git-test-track1-uncovered"

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

    predictions = trainer.predict(model, dataloaders=dm.test_dataloader())

    # Create an inverse mapping for the gloss vocabulary.
    # We use the training dataset gloss vocabulary (which was built solely from the training data).
    inv_gloss_vocab = {idx: token for token, idx in dm.train_dataset.gloss_vocab.items()}
    count = 0
    # Process and print predictions.
    print("\nPredictions on the test set:")
    for batch in predictions:
        # batch is a tensor of shape (batch_size, tgt_seq_len)
        for pred in batch:
            count+=1
            # Convert predicted indices to tokens; ignore padding.
            tokens = [inv_gloss_vocab.get(idx.item(), "<unk>") for idx in pred if idx.item() != gloss_pad_idx]
            # Truncate at the stop token "</s>" if present.
            if "</s>" in tokens:
                tokens = tokens[:tokens.index("</s>")]
            predicted_gloss = " ".join(tokens)
            print(f"Predicted Gloss {count}:", predicted_gloss)