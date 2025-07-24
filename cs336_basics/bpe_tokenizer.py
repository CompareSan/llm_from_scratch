import os
from collections import Counter
import regex as re
from typing import BinaryIO
from multiprocessing import Pool, cpu_count



class BPETokenizer:
    def __init__(
        self,
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        self.token_merges: list[tuple[bytes, bytes]] = []
        self.next_token_id = 256 + len(self.special_tokens)
        self._initialize_vocabulary()

    def _initialize_vocabulary(self) -> None:
        for i in range(256):
            self.vocab[i] = bytes([i])

        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode("utf-8")

    def train(
        self,
        input_path: str | os.PathLike,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        pretoken_counts = self._get_pretoken_counts(input_path, self.special_tokens)
        byte_pretoken_counts = self._convert_string_to_bytes(pretoken_counts)
        pair_counts = self._get_pair_counts(byte_pretoken_counts)
        self._train_loop(byte_pretoken_counts, pair_counts)
        return self.vocab, self.token_merges

    def _get_pretoken_counts(
        self,
        input_path: str | os.PathLike,
        special_tokens: list[str],
    ) -> Counter:
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, 100, special_tokens[0].encode("utf-8"))

            pretoken_counts: Counter = Counter()
            # Escape special tokens for regex
            escaped_tokens = list(map(re.escape, special_tokens))
            split_pattern = "|".join(escaped_tokens)

            # GPT-2-like pre-tokenization pattern
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

            tasks = [
                (input_path, start, end, split_pattern, PAT)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            with Pool(cpu_count()) as pool:
                results = pool.map(self._process_chunk, tasks)
        
            for counter in results:
                pretoken_counts.update(counter)

            return pretoken_counts

    def _process_chunk(self, args):
        """Process a single chunk and return its token counts."""
        file_path, start, end, split_pattern, PAT = args
        counter = Counter()
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            parts = re.split(split_pattern, chunk)
            for part in parts:
                for match in re.finditer(PAT, part):
                    token = match.group(0)
                    counter[token] += 1
        return counter

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

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

    def _convert_string_to_bytes(
        self,
        token_counts: Counter,
    ) -> Counter:
        """
        Converts string tokens to tuples of bytes for initial BPE vocabulary.
        """
        return Counter({tuple(token.encode("utf-8")): count for token, count in token_counts.items()})

    def _train_loop(self, byte_pretoken_counts: Counter, pair_counts: Counter) -> None:
        while self.next_token_id < self.vocab_size:
            most_frequent_pair = self._get_most_frequent_pair(pair_counts)
            self._create_new_token_and_update_vocab_and_merges(most_frequent_pair)
            pair_counts = self._update_pair_counts(pair_counts, most_frequent_pair, byte_pretoken_counts)
            byte_pretoken_counts = self._update_byte_pretoken_counts(byte_pretoken_counts, most_frequent_pair)
            self.next_token_id += 1

    def _get_pair_counts(
        self,
        byte_pretoken_counts: Counter,
    ) -> Counter:
        pair_counts: Counter = Counter()
        for seq, freq in byte_pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq
        return pair_counts

    def _get_most_frequent_pair(
        self,
        pair_counts: Counter,
    ) -> tuple[int, int]:
        most_frequent_pair, _ = max(pair_counts.items(), key=lambda x: (x[1],) + tuple(self.vocab[c] for c in x[0]))
        return most_frequent_pair

    def _create_new_token_and_update_vocab_and_merges(
        self,
        most_frequent_pair: tuple[int, int],
    ) -> None:
        id1, id2 = most_frequent_pair
        new_token_bytes = self.vocab[id1] + self.vocab[id2]
        self.vocab[self.next_token_id] = new_token_bytes
        self.token_merges.append((self.vocab[id1], self.vocab[id2]))

    def _update_byte_pretoken_counts(
        self,
        byte_pretoken_counts: Counter,
        most_frequent_pair: tuple[int, int],
    ) -> Counter:
        new_counts: Counter = Counter()
        id1, id2 = most_frequent_pair
        for seq, freq in byte_pretoken_counts.items():
            merged_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == id1 and seq[i + 1] == id2:
                    merged_seq.append(self.next_token_id)
                    i += 2
                else:
                    merged_seq.append(seq[i])
                    i += 1
            new_counts[tuple(merged_seq)] = freq
        return new_counts

    def _update_pair_counts(
        self,
        pair_counts: Counter,
        most_frequent_pair: tuple[int, int],
        byte_pretoken_counts: Counter,
    ):
        pair_counts.pop(most_frequent_pair)
        for key, count in list(byte_pretoken_counts.items()):
            key_len = len(key)
            for i in range(key_len - 2 + 1):
                if key[i : i + 2] == most_frequent_pair:
    
                    before = key[i - 1] if i > 0 else None
                    after = key[i + 2] if i + 2 < key_len else None

                    if before is not None:
                        pair_counts[(before, most_frequent_pair[0])] -= count
                    if after is not None:
                        pair_counts[(most_frequent_pair[-1], after)] -= count

                    if before is not None:
                        pair_counts[(before, self.next_token_id)] += count
                    if after is not None:
                        pair_counts[(self.next_token_id, after)] += count
        return pair_counts


if __name__ == "__main__":
    import json
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(vocab_size, special_tokens)
    vocab, token_merges = tokenizer.train(input_path="../data/TinyStoriesV2-GPT4-valid.txt")
    with open("vocab.json", "w") as vocab_file:
        json.dump({k: v.hex() for k, v in vocab.items()}, vocab_file, indent=2)
    with open("merges.txt", "w") as merges_file:
        for merge in token_merges:
            merges_file.write(f"{merge[0].hex()} {merge[1].hex()}\n")

    print("Vocabulary and merges saved.")
    print(vocab)
    print(token_merges)
