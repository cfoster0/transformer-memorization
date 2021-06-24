import functools
import torch
from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader
from secrets import token_bytes

def cycle(loader):
    while True:
        for data in loader:
            yield data

def tracked(dataset):

    class Wrapper(torch.utils.data.Dataset):
          
        def __init__(self, *args, **kwargs):
            self.num_calls = 0
            self.step_count = 0
            self.call_counters = Counter()
            self.calls = defaultdict(list)  
            self.dataset = dataset(*args, **kwargs)

        def __len__(self):
            return self.dataset.__len__()

        def __getitem__(self, idx):
            self.call_counters[idx] += 1
            self.calls[idx] += [self.step_count]
            self.num_calls += 1
            return self.dataset.__getitem__(idx)

        def step(self):
            self.step_count += 1

        def stats(self, idx):
            if self.calls[idx]:
                last_called = self.calls[idx][-1]
                staleness = self.step_count - last_called
            else:
                staleness = None

            return {
                'call_count': self.call_counters[idx], # How many times the datapoint has been accessed during training
                'calls': self.calls[idx], # List of training steps during which the datapoint was accessed
                'staleness': staleness, # Number of steps since the datapoint was last accessed
                'data': self.dataset.__getitem__(idx), # Data itself
            }
          
    return Wrapper

@tracked
class ArithmeticSequenceDataset(Dataset):
    """Arithmetic sequence sythetic dataset."""

    def __init__(self, difference, seqlen=128, limit=1_000, length=1_000_000):
        """
        Args:
            difference (int): Difference between subsequent items in sequence
            seqlen (int): Sequence length to use for generation
            limit (int): Maximum value 
            length (int): Number of sequences in dataset
        """
        self.difference = difference
        self.seqlen = seqlen
        self.limit = limit
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < self.size:
            bytes = " ".join([str(x) for x in range(idx, self.limit, self.difference)]).encode()[:self.seqlen]
            num_padding = self.seqlen - len(bytes)
            bytes += b' ' * num_padding
            item = torch.LongTensor([b for b in bytes])
            return item

        else:
            raise ValueError("Arithmetic sequence dataset indexed too far")

@tracked
class HashedIndexDataset(Dataset):
    """Hashed index sythetic dataset."""

    def __init__(self, seqlen=128, length=1_000_000):
        """
        Args:
            seqlen (int): Sequence length to use for generation
            length (int): Number of sequences in dataset
        """
        self.seqlen = seqlen
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < self.length:
            bytes = str(hash(idx)).encode()[:self.seqlen]
            num_padding = self.seqlen - len(bytes)
            bytes += b' ' * num_padding
            item = torch.LongTensor([b for b in bytes])
            return item
        else:
            raise ValueError("Hashed index dataset indexed too far")


@tracked
class RandomBytesDataset(Dataset):
    """Random bytes sythetic dataset."""

    def __init__(self, seqlen=128, length=1_000_000, inline_meta=False):
        """
        Args:
            seqlen (int): Sequence length to use for generation
            length (int): Number of sequences in dataset
            inline_metad (bool): Whether to include the datapoint index as inline metadata
        """
        self.seqlen = seqlen
        self.length = length
        self.inline_meta = inline_meta
        self.cache = {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < self.length:
            if idx in self.cache:
                return self.cache[idx]
            else:
                if self.inline_meta:
                    meta = str(idx).encode()
                    bytes = meta + token_bytes(self.seqlen - len(meta))
                else:
                    bytes = token_bytes(self.seqlen)
                item = torch.LongTensor([b for b in bytes])
                self.cache[idx] = item
                return item
        else:
            raise ValueError("Random bytes dataset indexed too far")