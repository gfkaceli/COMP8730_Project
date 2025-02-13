def minimum_edit_distance(source, target):
    """
    Calculate the minimum edit distance between two strings.

    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.

    Returns:
    int: The minimum edit distance between str1 and str2.
    """

    m, n = len(source), len(target)

    # create a matrix to store the distances
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Deleting all characters of str1
    for j in range(n + 1):
        dp[0][j] = j  # Inserting all characters into str1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:  # If characters match
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],    # Deletion
                                       dp[i][j - 1],    # Insertion
                                       dp[i - 1][j - 1]) # Substitution

    return dp[m][n]

