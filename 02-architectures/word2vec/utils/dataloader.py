import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset
from utils.constants import CBOW_N_WORDS, SKIPGRAM_N_WORDS, MIN_WORD_FREQUENCY, MAX_SEQUENCE_LENGTH

# Directory and file mapping for caching splits

SPLIT_FILES = {
    'train': 'train.txt',
    'validation': 'validation.txt',
    'test': 'test.txt'
}


def cache_split(split: str, data_dir) -> str:
    file_name = SPLIT_FILES[split]
    file_path = os.path.join(data_dir, file_name)
    print(file_path)
    if not os.path.exists(file_path):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading dataset...")
        ds = load_dataset("wikitext", "wikitext-103-v1", split=split)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in ds:
                # collapse newlines into spaces for consistency
                text = item["text"].strip().replace("\n", " ")
                if text:
                    f.write(text + "\n")
        print("Download completed!")
    return file_path

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

def build_vocab(data_dir) -> dict:
    train_file = cache_split('train', data_dir)
    counter = Counter()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = tokenize(line)
            counter.update(tokens)
   
    tokens = [tok for tok, freq in counter.items() if freq >= MIN_WORD_FREQUENCY]
    tokens = ['<unk>'] + tokens
    vocab = {tok: idx for idx, tok in enumerate(tokens)}
    return vocab

class Word2VecDataset(Dataset):
    """
    CBOW or Skip-Gram dataset for WikiText-103, with local caching.
    """
    def __init__(self, split: str, vocab: dict, data_dir:str, model_type: str = 'cbow'):
        self.vocab = vocab
        self.data_dir = data_dir
        self.model_type = model_type.lower()
        self.context_size = CBOW_N_WORDS if self.model_type == 'cbow' else SKIPGRAM_N_WORDS
        self.unk_idx = vocab.get('<unk>')

        # Ensure the split is cached, then read lines
        file_path = cache_split(split, data_dir)
        tokens = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_tokens = tokenize(line)
                tokens.extend([vocab.get(tok, self.unk_idx) for tok in line_tokens])
                if MAX_SEQUENCE_LENGTH and len(tokens) >= MAX_SEQUENCE_LENGTH:
                    tokens = tokens[:MAX_SEQUENCE_LENGTH]
                    break

        # Build (input, target) pairs
        self.pairs = []
        for i in range(self.context_size, len(tokens) - self.context_size):
            center = tokens[i]
            context = [tokens[i + offset]
                       for offset in range(-self.context_size, self.context_size + 1)
                       if offset != 0]
            if self.model_type == 'cbow':
                self.pairs.append((context, center))
            else:
                for ctx in context:
                    self.pairs.append((center, ctx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# Collate functions for batching
def cbow_collate(batch):
    contexts, targets = zip(*batch)
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def skipgram_collate(batch):
    centers, targets = zip(*batch)
    return torch.tensor(centers, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def get_dataloader_and_vocab(train_batch_size: int = 96, val_batch_size=96,
                             model_name: str = 'cbow', data_dir='../data/wikitext-103-v1'):
    """
    Returns:
      train_loader, val_loader, vocab
    """
    vocab = build_vocab(data_dir)
    train_ds = Word2VecDataset('train', vocab, data_dir, model_name)
    val_ds = Word2VecDataset('validation', vocab, data_dir, model_name)

    collate_fn = cbow_collate if model_name.lower() == 'cbow' else skipgram_collate

    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, vocab
