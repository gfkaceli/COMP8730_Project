"""
my_data.py

This module loads glossing data from files in the custom format:
    \t <source sentence>
    \g <gloss>
    \l <translation>

Each sample is separated by a blank line. This module provides a custom Dataset class and
functions that return DataLoader objects for training, validation, and testing.
"""

from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional


def read_glossing_file_custom(file_path: str) -> List[Dict[str, Optional[str]]]:
    """
    Reads a glossing file with the following format:
      - Lines starting with "\t" contain the source sentence.
      - Lines starting with "\g" contain the gloss.
      - Lines starting with "\l" contain the translation.
    Each sample is separated by a blank line.

    Args:
        file_path (str): Path to the data file.

    Returns:
        List[Dict[str, Optional[str]]]: A list of samples, each sample is a dictionary with keys:
            "source", "gloss", and "translation".
    """
    samples = []
    current_sample = {"source": None, "gloss": None, "translation": None}

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Blank line indicates end of current sample.
            if not line:
                if any(current_sample.values()):
                    samples.append(current_sample)
                current_sample = {"source": None, "gloss": None, "translation": None}
                continue
            if line.startswith("\\t"):
                current_sample["source"] = line[2:].strip()
            elif line.startswith("\\g"):
                current_sample["gloss"] = line[2:].strip()
            elif line.startswith("\\l"):
                current_sample["translation"] = line[2:].strip()
            else:
                continue
    # Append the last sample if file does not end with a blank line.
    if any(current_sample.values()):
        samples.append(current_sample)
    return samples


class CustomGlossingDataset(Dataset):
    """
    Custom Dataset for glossing data.
    Each sample is a dictionary with keys: "source", "gloss", "translation".
    """

    def __init__(self, file_path: str):
        self.samples = read_glossing_file_custom(file_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Optional[str]]:
        return self.samples[idx]


def my_collate_fn(batch: List[Dict[str, Optional[str]]]) -> Dict[str, List[str]]:
    """
    Collate function that aggregates a list of sample dictionaries into a single batch dictionary.
    Each key (source, gloss, translation) maps to a list of strings.
    """
    # Ensure that each sample has all keys.
    sources = [sample.get("source", "") for sample in batch]
    glosses = [sample.get("gloss", "") for sample in batch]
    translations = [sample.get("translation", "") for sample in batch]
    return {"source": sources, "gloss": glosses, "translation": translations}


def get_train_dataloader(train_file: str, batch_size: int = 32) -> DataLoader:
    """
    Loads training data from a file and returns a DataLoader object.

    Args:
        train_file (str): Path to the training data file.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object for training data.
    """
    dataset = CustomGlossingDataset(train_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    return dataloader


def get_validation_dataloader(val_file: str, batch_size: int = 32) -> DataLoader:
    """
    Loads validation data from a file and returns a DataLoader object.

    Args:
        val_file (str): Path to the validation data file.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object for validation data.
    """
    dataset = CustomGlossingDataset(val_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
    return dataloader


def get_test_dataloader(test_file: str, batch_size: int = 32) -> DataLoader:
    """
    Loads test data from a file and returns a DataLoader object.

    Args:
        test_file (str): Path to the test data file.
        batch_size (int): Batch size.

    Returns:
        DataLoader: DataLoader object for test data.
    """
    dataset = CustomGlossingDataset(test_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=my_collate_fn)
    return dataloader


# run for logging purposes
if __name__ == "__main__":
    # Example usage:
    train_file = "data/Lezgi/lez-train-track1-uncovered"
    val_file = "data/Lezgi/lez-dev-track1-uncovered"
    test_file = "data/Lezgi/lez-test-track1-uncovered"

    train_loader = get_train_dataloader(train_file, batch_size=32)
    val_loader = get_validation_dataloader(val_file, batch_size=32)
    test_loader = get_test_dataloader(test_file, batch_size=32)

    print("Number of training samples:", len(train_loader.dataset))
    print("Number of validation samples:", len(val_loader.dataset))
    print("Number of test samples:", len(test_loader.dataset))

    # Print a single batch sample
    for batch in train_loader:
        print("Batch sample:")
        print(batch)
        break
