import os
from typing import BinaryIO
import regex as re 
from collections import Counter


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_pretoken_counts(input_path: str | os.PathLike, special_tokens: list[str]) -> Counter:
    """
    Reads a text file and counts the occurrences of pre-tokens using regex-based pre-tokenization.
    """
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, 100, special_tokens[0].encode("utf-8")
        )

        pretoken_counts: Counter = Counter()
        # Escape special tokens for regex
        escaped_tokens = list(map(re.escape, special_tokens))
        split_pattern = "|".join(escaped_tokens)

        # GPT-2-like pre-tokenization pattern
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Keep special tokens in the split result
            # I don't want to keep them
            parts = re.split(split_pattern, chunk)

            for part in parts:
                for match in re.finditer(PAT, part):
                    token = match.group(0)
                    pretoken_counts[token] += 1

    return pretoken_counts


def get_dictionary_of_bytes(token_counts: Counter) -> Counter:
    """
    Converts string tokens to tuples of bytes for initial BPE vocabulary.
    """
    return Counter({
        tuple(token.encode("utf-8")): count
        for token, count in token_counts.items()
    })


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