import os
from collections import Counter
from cs336_basics.pretokenizer import Pretokenizer


class BPETrainer:
    def __init__(
        self,
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab: dict[int, bytes] = {}
        self.token_merges: list[tuple[bytes, bytes]] = []
        self._next_token_id = 256 + len(special_tokens)
        self._initialize_vocabulary()
        self.pretokenizer = Pretokenizer(special_tokens)

    def _initialize_vocabulary(self) -> None:
        for i in range(256):
            self.vocab[i] = bytes([i])

        for i, token in enumerate(self.special_tokens, start=256):
            self.vocab[i] = token.encode("utf-8")

    def train(
        self,
        input_path: str | os.PathLike,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        byte_pretoken_counts = self.pretokenizer._get_pretoken_counts(input_path)
        pair_byte_counts = self._get_pair_byte_counts(byte_pretoken_counts)
        self._train_loop(byte_pretoken_counts, pair_byte_counts)
        return self.vocab, self.token_merges

    def _train_loop(self, byte_pretoken_counts: Counter, pair_byte_counts: Counter) -> None:
        while self._next_token_id < self.vocab_size:
            most_frequent_pair = self._get_most_frequent_pair(pair_byte_counts)
            self._create_new_token_and_update_vocab_and_merges(most_frequent_pair)
            pair_byte_counts = self._update_pair_byte_counts(pair_byte_counts, most_frequent_pair, byte_pretoken_counts)
            byte_pretoken_counts = self._update_byte_pretoken_counts(byte_pretoken_counts, most_frequent_pair)
            self._next_token_id += 1

    def _get_pair_byte_counts(
        self,
        byte_pretoken_counts: Counter,
    ) -> Counter:
        pair_byte_counts: Counter = Counter() # use a priority queue for efficiency
        for seq, freq in byte_pretoken_counts.items():
            for i in range(len(seq) - 1):
                pair_byte_counts[(seq[i], seq[i + 1])] += freq
        return pair_byte_counts

    def _get_most_frequent_pair(
        self,
        pair_byte_counts: Counter,
    ) -> tuple[int, int]: # use a heapq.pop for efficiency
        most_frequent_pair, _ = max(pair_byte_counts.items(), key=lambda x: (x[1],) + tuple(self.vocab[c] for c in x[0]))
        return most_frequent_pair

    def _create_new_token_and_update_vocab_and_merges(
        self,
        most_frequent_pair: tuple[int, int],
    ) -> None:
        id1, id2 = most_frequent_pair
        new_token_bytes = self.vocab[id1] + self.vocab[id2]
        self.vocab[self._next_token_id] = new_token_bytes
        self.token_merges.append((self.vocab[id1], self.vocab[id2]))

    def _update_byte_pretoken_counts(
        self,
        byte_pretoken_counts: Counter,
        most_frequent_pair: tuple[int, int],
    ) -> Counter:
        id1, id2 = most_frequent_pair
        for seq, freq in list(byte_pretoken_counts.items()):
            merged_seq = []
            i = 0
            changed = False
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == id1 and seq[i + 1] == id2:
                    merged_seq.append(self._next_token_id)
                    i += 2
                    changed = True
                else:
                    merged_seq.append(seq[i])
                    i += 1
            if changed:
                byte_pretoken_counts.pop(seq)
                byte_pretoken_counts[tuple(merged_seq)] = freq
        return byte_pretoken_counts

    def _update_pair_byte_counts(
        self,
        pair_byte_counts: Counter,
        most_frequent_pair: tuple[int, int],
        byte_pretoken_counts: Counter,
    ):
        pair_byte_counts.pop(most_frequent_pair)
        for key, count in list(byte_pretoken_counts.items()):
            key_len = len(key)
            for i in range(key_len - 2 + 1):
                if key[i : i + 2] == most_frequent_pair:
    
                    before = key[i - 1] if i > 0 else None
                    after = key[i + 2] if i + 2 < key_len else None

                    if before is not None:
                        pair_byte_counts[(before, most_frequent_pair[0])] -= count
                    if after is not None:
                        pair_byte_counts[(most_frequent_pair[-1], after)] -= count

                    if before is not None:
                        pair_byte_counts[(before, self._next_token_id)] += count
                    if after is not None:
                        pair_byte_counts[(self._next_token_id, after)] += count
        return pair_byte_counts


if __name__ == "__main__":
    import json
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    bpe_trainer = BPETrainer(vocab_size, special_tokens)
    vocab, token_merges = bpe_trainer.train(input_path="../data/TinyStoriesV2-GPT4-train.txt")
    with open("vocab.json", "w") as vocab_file:
        json.dump({k: v.hex() for k, v in vocab.items()}, vocab_file, indent=2)
    with open("merges.txt", "w") as merges_file:
        for merge in token_merges:
            merges_file.write(f"{merge[0].hex()} {merge[1].hex()}\n")

    print("Vocabulary and merges saved.")
    print(vocab)
    print(token_merges)
