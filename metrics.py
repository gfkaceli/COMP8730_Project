
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


def average_word_edit_distance(predictions: list[str], targets: list[str]) -> float:
    """
    Computes the average Levenshtein edit distance between lists of predicted glosses
    and target glosses.

    Each prediction and corresponding target is assumed to be a string.
    The edit distance for each pair is computed using the word_edit_distance function,
    and the average edit distance over all pairs is returned.

    Args:
        predictions (list of str): List of predicted gloss strings.
        targets (list of str): List of target (gold) gloss strings.

    Returns:
        float: The average edit distance over the dataset.

    Raises:
        ValueError: If the lengths of the prediction and target lists do not match.
    """
    if len(predictions) != len(targets):
        raise ValueError("The number of predictions and targets must be the same.")

    total_distance = 0
    for pred, target in zip(predictions, targets):
        total_distance += word_edit_distance(pred, target)

    return total_distance / len(predictions)


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