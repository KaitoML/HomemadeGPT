class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _find_most_frequent_pair(indices):
        counts = {}

        for pair in zip(indices, indices[1:]):
            counts[pair] = counts.get(pair, 0) + 1

        return max(counts, key=counts.get)

    @staticmethod
    def _merge_pair(indices, pair, idx):

        new_indices = []
        i = 0

        while i < len(indices):

            if i < len(indices) - 1 and (indices[i], indices[i + 1]) == pair:
                new_indices.append(idx)
                i += 2
            else:
                new_indices.append(indices[i])
                i += 1

        return new_indices

    def train(self, text, num_merges=1000):
        from tqdm.auto import tqdm

        tokens = list(text.encode('utf-8'))
        idx = 256

        loop = tqdm(range(num_merges), total=num_merges, desc='Merging tokens')
        for _ in loop:

            if len(tokens) < 2:
                break

            pair = self._find_most_frequent_pair(tokens)
            tokens = self._merge_pair(tokens, pair, idx)
            self.merges[pair] = idx
            idx += 1

        for (a, b), new_idx in self.merges.items():
            self.vocab[new_idx] = self.vocab[a] + self.vocab[b] #

        return tokens

    def encode(self, text): # built with AI assistance
        tokens = list(text.encode('utf-8'))

        while len(tokens) >= 2: # i.e. while there is at least one pair to merge
            pairs = set(zip(tokens, tokens[1:])) # all unique consecutive characters
            candidates = [p for p in pairs if p in self.merges] # all pairs that occurred in merges

            if not candidates:
                break

            pair = min(candidates, key=lambda p: self.merges[p]) # the pair with the smallest index in merges, i.e. the earliest merge
            tokens = self._merge_pair(tokens, pair, self.merges[pair]) # apply merge

        return tokens


    def decode(self, indices):
        token_bytes = b"".join(self.vocab[i] for i in indices)
        return token_bytes.decode('utf-8', errors='replace')