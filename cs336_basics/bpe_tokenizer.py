import os
from collections import Counter
from cs336_basics.pretokenization_example import get_pretoken_counts, convert_string_to_bytes


class BPETokenizer():
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        self.token_merges: list[tuple[bytes, bytes]] = []
        self.next_token_id = 256 + len(self.special_tokens)
        self._initialize_vocabulary()
    
    def _initialize_vocabulary(self) -> None:
        # Base vocab
        for i in range(256):
            self.vocab[i] = bytes([i])

        # Add special tokens
        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode("utf-8")
    
    def train(self, input_path: str | os.PathLike):
        pretoken_counts = get_pretoken_counts(input_path, self.special_tokens)
        byte_pretoken_counts = convert_string_to_bytes(pretoken_counts)
        while self.next_token_id < self.vocab_size:
            pair_counts = self._get_pair_counts(byte_pretoken_counts)
            most_frequent_pair = self._get_most_frequent_pair(pair_counts)
            self._create_new_token_and_update_vocab_and_merges(most_frequent_pair)
            byte_pretoken_counts = self._update_byte_pretoken_counts(byte_pretoken_counts, most_frequent_pair)
            self.next_token_id += 1
        
        return self.vocab, self.token_merges


    def _get_pair_counts(self, byte_pretoken_counts: Counter) -> Counter:
        pair_counts: Counter = Counter()
        for seq, freq in byte_pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair_counts[(seq[i], seq[i + 1])] += freq
        return pair_counts
    
    def _get_most_frequent_pair(self, pair_counts: Counter) -> tuple[int, int]:
        most_frequent_pair, _ = max(
            pair_counts.items(),
            key=lambda x: (x[1],) + tuple(self.vocab[c] for c in x[0])
        )
        return most_frequent_pair
    
    def _create_new_token_and_update_vocab_and_merges(self, most_frequent_pair: tuple[int, int]) -> None:
        id1, id2 = most_frequent_pair
        new_token_bytes = self.vocab[id1] + self.vocab[id2]
        self.vocab[self.next_token_id] = new_token_bytes
        self.token_merges.append((self.vocab[id1], self.vocab[id2]))
    
    def _update_byte_pretoken_counts(self, byte_pretoken_counts: Counter, most_frequent_pair: tuple[int, int]) -> Counter:
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
            byte_pretoken_counts = new_counts
        return byte_pretoken_counts



if __name__ == "__main__":
    vocab_size = 270
    special_tokens = ["<|endoftext|>"]
    tokenizer = BPETokenizer(vocab_size, special_tokens)
    vocab, token_merges = tokenizer.train(input_path="../data/corpus.en")
    print(vocab)
    print(token_merges)