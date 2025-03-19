"""
data.py

This module loads glossing data from a plain-text file with the following format:
    \t <source sentence>
    \g <gloss>
    \l <translation>

Each sample is separated by a blank line.
It provides:
  - A custom Dataset class (GlossingDataset) that reads the file, builds vocabularies (if not provided),
    and converts text to padded tensors (including computing source lengths).
  - A collate function that stacks samples into a batch.
  - A LightningDataModule (GlossingDataModule) that returns DataLoader objects for training, validation, and testing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List, Dict, Optional
from pytorch_lightning import LightningDataModule
from torchtext.vocab import build_vocab_from_iterator

# Define special tokens.
SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>"]


class GlossingDataset(Dataset):
    def __init__(self, file_path: str, max_src_len: int = 100, max_tgt_len: int = 30, max_trans_len: int = 50,
                 src_vocab: Optional[Dict[str, int]] = None,
                 gloss_vocab: Optional[Dict[str, int]] = None,
                 trans_vocab: Optional[Dict[str, int]] = None):
        """
        Reads glossing data from a file. Each sample in the file is separated by a blank line and is expected to contain:
            \t <source sentence>
            \g <gloss>
            \l <translation>

        The source sentence is tokenized at the character level (via flattening),
        while gloss and translation are tokenized by whitespace.

        Args:
            file_path (str): Path to the data file.
            max_src_len (int): Maximum source length (in characters when char_level=True).
            max_tgt_len (int): Maximum gloss length.
            max_trans_len (int): Maximum translation length.
            src_vocab (Optional[Dict[str,int]]): Pre-built vocabulary for source tokens (character-level).
            gloss_vocab (Optional[Dict[str,int]]): Pre-built vocabulary for gloss tokens.
            trans_vocab (Optional[Dict[str,int]]): Pre-built vocabulary for translation tokens.
        """
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len
        self.samples = self.read_file(file_path)

        # If vocabularies are not provided, build them from the data.
        self.src_vocab = src_vocab if src_vocab is not None else self.build_vocab([s["source"] for s in self.samples],
                                                                                  char_level=True)
        self.gloss_vocab = gloss_vocab if gloss_vocab is not None else self.build_vocab(
            [s["gloss"] for s in self.samples], char_level=False)
        self.trans_vocab = trans_vocab if trans_vocab is not None else self.build_vocab(
            [s["translation"] for s in self.samples], char_level=False)

        # Ensure special tokens exist in the gloss vocabulary.
        for token in SPECIAL_TOKENS:
            if token not in self.gloss_vocab:
                self.gloss_vocab[token] = len(self.gloss_vocab)

    def read_file(self, file_path: str) -> List[Dict[str, Optional[List[str]]]]:
        samples = []
        current_sample = {"source": None, "gloss": None, "translation": None}
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if any(current_sample.values()):
                        samples.append(current_sample)
                    current_sample = {"source": None, "gloss": None, "translation": None}
                    continue
                if line.startswith("\\t"):
                    current_sample["source"] = line[2:].strip().split()
                elif line.startswith("\\g"):
                    current_sample["gloss"] = line[2:].strip().split()
                elif line.startswith("\\l"):
                    current_sample["translation"] = line[2:].strip().split()
                else:
                    continue
            if any(current_sample.values()):
                samples.append(current_sample)
        return samples

    def build_vocab(self, texts: List[List[str]], char_level: bool = False) -> Dict[str, int]:
        counter = Counter()
        for tokens in texts:
            if tokens is None:
                continue
            if char_level:
                for token in tokens:
                    counter.update(list(token))
            else:
                counter.update(tokens)
        # Sort tokens to ensure contiguous indices.
        sorted_tokens = sorted(counter.keys())
        # Reserve index 0 for <pad> and 1 for <unk>, then assign contiguous indices starting at 2.
        vocab = {tok: i for i, tok in enumerate(sorted_tokens, start=2)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        return vocab

    def text_to_tensor(self, tokens: List[str], vocab: Dict[str, int], max_len: int,
                       char_level: bool = False) -> torch.Tensor:
        if char_level:
            # For source, if char_level is True, flatten tokens into individual characters.
            chars = []
            for token in tokens:
                chars.extend(list(token))
            tokens = chars
        indices = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
        indices += [vocab["<pad>"]] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def tensor_to_text(self, tensor: torch.Tensor, vocab: Dict[str, int]) -> str:
        inv_vocab = {idx: tok for tok, idx in vocab.items()}
        tokens = [inv_vocab.get(idx.item(), "<unk>") for idx in tensor if idx.item() != vocab["<pad>"]]
        return " ".join(tokens)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        src_tokens = sample["source"]
        gloss_tokens = sample["gloss"]
        trans_tokens = sample["translation"]

        # For source, flatten tokens into characters.
        flattened_src = []
        for token in src_tokens:
            flattened_src.extend(list(token))
        src_tensor = self.text_to_tensor(flattened_src, self.src_vocab, self.max_src_len, char_level=False)
        # For gloss, add start and end tokens.
        gloss_tokens = ["<s>"] + gloss_tokens + ["</s>"]
        gloss_tensor = self.text_to_tensor(gloss_tokens, self.gloss_vocab, self.max_tgt_len, char_level=False)
        trans_tensor = self.text_to_tensor(trans_tokens, self.trans_vocab, self.max_trans_len, char_level=False)

        # Compute source length as the number of characters after flattening.
        src_len = min(len(flattened_src), self.max_src_len)
        return src_tensor, src_len, gloss_tensor, trans_tensor


def collate_fn(batch: List) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
    src_list, src_len_list, gloss_list, trans_list = zip(*batch)
    src = torch.stack(src_list, dim=0)
    src_lengths = torch.tensor(src_len_list, dtype=torch.long)
    gloss = torch.stack(gloss_list, dim=0)
    trans = torch.stack(trans_list, dim=0)
    return src, src_lengths, gloss, trans


class GlossingDataModule(LightningDataModule):
    def __init__(self, train_file: str, val_file: str, test_file: str, batch_size: int = 32,
                 max_src_len: int = 100, max_tgt_len: int = 30, max_trans_len: int = 50):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_trans_len = max_trans_len

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # Build training dataset.
            self.train_dataset = GlossingDataset(self.train_file, self.max_src_len, self.max_tgt_len,
                                                 self.max_trans_len)
            # Build validation dataset using training vocabularies.
            self.val_dataset = GlossingDataset(
                self.val_file, self.max_src_len, self.max_tgt_len, self.max_trans_len,
                src_vocab=self.train_dataset.src_vocab,
                gloss_vocab=self.train_dataset.gloss_vocab,
                trans_vocab=self.train_dataset.trans_vocab
            )
            self.src_vocab = self.train_dataset.src_vocab
            self.gloss_vocab = self.train_dataset.gloss_vocab
            self.trans_vocab = self.train_dataset.trans_vocab
            self.source_alphabet_size = len(self.src_vocab)
            self.target_alphabet_size = len(self.gloss_vocab)
            self.trans_alphabet_size = len(self.trans_vocab)
            # Build tokenizers using torchtext's build_vocab_from_iterator.
            self.source_tokenizer = build_vocab_from_iterator([[token] for token in sorted(self.src_vocab.keys())],
                                                              specials=SPECIAL_TOKENS)
            self.target_tokenizer = build_vocab_from_iterator([[token] for token in sorted(self.gloss_vocab.keys())],
                                                              specials=SPECIAL_TOKENS)
            self.source_tokenizer.set_default_index(self.source_tokenizer["<unk>"])
            self.target_tokenizer.set_default_index(self.target_tokenizer["<unk>"])
            self._batch_collate = collate_fn
        if stage == "test" or stage is None:
            self.test_dataset = GlossingDataset(
                self.test_file, self.max_src_len, self.max_tgt_len, self.max_trans_len,
                src_vocab=self.train_dataset.src_vocab,
                gloss_vocab=self.train_dataset.gloss_vocab,
                trans_vocab=self.train_dataset.trans_vocab
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._batch_collate, num_workers=6, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._batch_collate, num_workers=6, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self._batch_collate, num_workers=6, persistent_workers=True)


if __name__ == "__main__":
    # Example usage.
    train_file = "data/Lezgi/lez-train-track1-uncovered"
    val_file = "data/Lezgi/lez-dev-track1-uncovered"
    test_file = "data/Lezgi/lez-test-track1-uncovered"

    dm = GlossingDataModule(train_file, val_file, test_file, batch_size=32)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    print("Number of training samples:", len(dm.train_dataset))
    print("Number of validation samples:", len(dm.val_dataset))
    print("Number of test samples:", len(dm.test_dataset))

    # Print a single batch sample in text format.
    loader = dm.test_dataloader()
    for batch in loader:
        src_tensor, src_lengths, gloss_tensor, trans_tensor = batch
        print("Batch sample (in text format):")
        for i in range(src_tensor.size(0)):
            src_text = dm.test_dataset.tensor_to_text(src_tensor[i], dm.test_dataset.src_vocab)
            gloss_text = dm.test_dataset.tensor_to_text(gloss_tensor[i], dm.test_dataset.gloss_vocab)
            trans_text = dm.test_dataset.tensor_to_text(trans_tensor[i], dm.test_dataset.trans_vocab)
            print(f"Sample {i + 1}:")
            print("  Source:", src_text)
            print("  Gloss: ", gloss_text)
            print("  Trans: ", trans_text)
        break
