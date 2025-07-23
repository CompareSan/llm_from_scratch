import os
from collections import Counter
from cs336_basics.pretokenization_example import get_pretoken_counts, get_dictionary_of_bytes

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a Byte Pair Encoding (BPE) tokenizer from scratch.

    Args:
        input_path: Path to the input text file.
        vocab_size: Desired vocabulary size.
        special_tokens: List of special tokens (strings).

    Returns:
        vocab: Dictionary mapping token IDs to bytes.
        token_merges: List of merges in the order they were applied.
    """
    # Step 1: Pre-tokenize and count
    pretoken_counts = get_pretoken_counts(input_path, special_tokens)
    byte_token_counts = get_dictionary_of_bytes(pretoken_counts)

    # Initialize vocab and merges
    vocab: dict[int, bytes] = {}
    token_merges: list[tuple[bytes, bytes]] = []

    # Base vocab: single-byte tokens
    for i in range(256):
        vocab[i] = bytes([i])

    # Add special tokens
    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")

    next_token_id = 256 + len(special_tokens)

    # Step 2: Merge loop
    while next_token_id < vocab_size:
        # Count adjacent token pairs
        pair_counts: Counter = Counter()
        for seq, freq in byte_token_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq

        if not pair_counts:
            break  # No pairs left to merge

        # Pick most frequent pair; tie-break by lexicographically greater combined string
        (best_pair, _) = max(
            pair_counts.items(),
            key=lambda x: (x[1],) + tuple(vocab[c] for c in x[0])
        )
        id1, id2 = best_pair

        # Create new token
        new_token_bytes = vocab[id1] + vocab[id2]
        vocab[next_token_id] = new_token_bytes

        # Record merge
        token_merges.append((vocab[id1], vocab[id2]))

        # Update sequences with new token
        new_counts: Counter = Counter()
        for seq, freq in byte_token_counts.items():
            merged_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == id1 and seq[i + 1] == id2:
                    merged_seq.append(next_token_id)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_counts[tuple(merged_seq)] = freq

        byte_token_counts = new_counts
        next_token_id += 1

    return vocab, token_merges


if __name__ == "__main__":
    vocab_size = 270
    special_tokens = ["<|endoftext|>"]
    vocab, token_merges = train_bpe(input_path="../data/corpus.en",
              vocab_size=vocab_size,
              special_tokens=special_tokens)
    print(vocab)
    print(token_merges)