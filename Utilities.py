"""Extra functions for utility and convenience to be incorporated into our pipeline"""

import torch

def make_mask(seq_len, lengths):
    """
    Creates a boolean mask for a batch of sequences.
    Args:
        seq_len (int): Maximum sequence length.
        lengths (Tensor): Tensor of shape (batch,) with valid lengths.
    Returns:
        mask (Tensor): Boolean mask of shape (batch, seq_len) where True indicates a valid position.
    """
    batch_size = lengths.size(0)
    mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0).expand(batch_size, seq_len)
    return mask < lengths.unsqueeze(1)

def make_mask_2d(lengths):
    """
    Given a tensor of lengths, returns a 2D mask (batch x max_length).
    """
    max_len = lengths.max().item()
    return make_mask(max_len, lengths)

def make_mask_3d(word_lengths, num_morphemes):
    batch_size = word_lengths.size(0)
    max_len = word_lengths.max().item()
    # Use 1 instead of max_morphemes so that the mask shape is (batch_size, max_len, 1)
    return torch.zeros(batch_size, max_len, 1, dtype=torch.bool, device=word_lengths.device)

def max_pool_2d(x: torch.Tensor, lengths: torch.Tensor):
    # x: shape [batch x timesteps x features]
    mask = make_mask_2d(lengths).to(x.device).unsqueeze(-1)
    x = torch.masked_fill(x, mask=mask, value=-1e9)
    x = torch.max(x, dim=1).values
    return x


def aggregate_segments(encoder_outputs: torch.Tensor, segmentation_mask: torch.Tensor) -> torch.Tensor:
    """
    Aggregates encoder outputs into morpheme-level representations using segmentation boundaries.

    Args:
        encoder_outputs: Tensor of shape (batch_size, seq_len, embed_dim) containing encoder outputs.
        segmentation_mask: Tensor of shape (batch_size, seq_len) with binary values (1 indicates a boundary).

    Returns:
        seg_tensor: Tensor of shape (batch_size, max_segments, embed_dim) containing averaged morpheme representations.
    """
    batch_size, seq_len, embed_dim = encoder_outputs.size()
    segments = []  # List to store segments for each word in the batch
    num_segments_list = []

    for b in range(batch_size):
        word_enc = encoder_outputs[b]  # (seq_len, embed_dim)
        seg_mask = segmentation_mask[b]  # (seq_len,)
        seg_reps = []
        start = 0
        for i in range(seq_len):
            # Check if current character is a boundary
            if seg_mask[i] >= 0.5:
                # Aggregate characters from start to i (inclusive)
                if i >= start:  # Ensure non-empty segment
                    seg_rep = word_enc[start:i + 1].mean(dim=0)
                    seg_reps.append(seg_rep)
                start = i + 1
        # If any characters remain after the last boundary, aggregate them
        if start < seq_len:
            seg_rep = word_enc[start:seq_len].mean(dim=0)
            seg_reps.append(seg_rep)
        # If no boundaries were detected, fall back to a single segment (average of entire word)
        if len(seg_reps) == 0:
            seg_reps.append(word_enc.mean(dim=0))
        seg_reps = torch.stack(seg_reps, dim=0)  # (num_segments, embed_dim)
        segments.append(seg_reps)
        num_segments_list.append(seg_reps.size(0))

    # Pad all segment tensors to the maximum number of segments in the batch.
    max_segments = max(num_segments_list)
    seg_tensor = torch.zeros(batch_size, max_segments, embed_dim, device=encoder_outputs.device)
    for b in range(batch_size):
        segs = segments[b]
        seg_tensor[b, :segs.size(0), :] = segs
    return seg_tensor


def word_edit_distance(predicted: str, target: str) -> int:
    """
    Computes the Levenshtein edit distance between two sequences of words.
    Both predicted and target should be strings. The function splits them
    into words (using whitespace) and computes the minimum number of
    insertions, deletions, and substitutions required to transform the
    predicted word list into the target word list.
    """
    pred_words = predicted.split()
    target_words = target.split()
    n = len(pred_words)
    m = len(target_words)

    # Initialize DP table.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1])  # substitution
    return dp[n][m]


def compute_word_level_gloss_accuracy(predictions: list, targets: list) -> float:
    """
    Computes word-level glossing accuracy over a set of predictions.
    A predicted gloss is considered correct only if it exactly matches the
    corresponding target gloss (after trimming).

    Args:
        predictions (list of str): List of predicted gloss strings.
        targets (list of str): List of ground truth gloss strings.

    Returns:
        float: Fraction of glosses that exactly match the target.
    """
    if len(targets) == 0:
        return 1.0
    correct = sum(1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip())
    return correct / len(targets)


def compute_morpheme_level_gloss_accuracy(predictions: list, targets: list, pad_token: str = "NULL") -> float:
    """
    Computes morpheme-level glossing accuracy over a set of predictions.
    For each predicted gloss, if the number of tokens is less than the target,
    the prediction is padded with the specified pad_token. If there are extra tokens,
    they are discarded. Accuracy is defined as the fraction of morphemes that are correctly
    glossed.

    Args:
        predictions (list of str): List of predicted gloss strings.
        targets (list of str): List of ground truth gloss strings.
        pad_token (str): Token used to pad predictions if too few morphemes are predicted.

    Returns:
        float: Fraction of correctly glossed morphemes over all morphemes.
    """
    total_correct = 0
    total_tokens = 0
    for pred, target in zip(predictions, targets):
        pred_tokens = pred.split()
        target_tokens = target.split()
        # Pad or truncate predicted tokens to match target length.
        if len(pred_tokens) < len(target_tokens):
            pred_tokens += [pad_token] * (len(target_tokens) - len(pred_tokens))
        elif len(pred_tokens) > len(target_tokens):
            pred_tokens = pred_tokens[:len(target_tokens)]
        # Compare tokens.
        correct = sum(1 for p, t in zip(pred_tokens, target_tokens) if p == t)
        total_correct += correct
        total_tokens += len(target_tokens)
    return total_correct / total_tokens if total_tokens > 0 else 1.0