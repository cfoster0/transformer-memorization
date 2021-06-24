from transformer_memorization import RandomBytesDataset

# This code was adapted from lucidrains existing `x-transformers` repository.
from simple_parallel_transformer import Transformer, Config
from simple_parallel_transformer.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import hydra

import time
import wandb

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class RandomBytesConfig:
    """Class for keeping track of config variables."""
    dataset_size: int
    inline_meta: bool
    depth: int
    heads: int
    head_dim: int
    vocab_size: int
    max_seq_len: int = 2048
    expansion_factor: int = 4
    max_seq_len: int = 1024
    shuffle: bool = True
    num_batches: int = int(1e5)
    batch_size: int = 4
    gradient_accumulate_every: int = 4
    learning_rate: float = 1e-4
    validate_every: int = 100
    generate_every: int = 500
    generate_legnth: int = 512
        
cs = ConfigStore.instance()
cs.store(name="random-bytes-config", node=RandomBytesConfig)

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))



@hydra.main(config_path=None, config_name="random-bytes-config")
def train(cfg: RandomBytesConfig) -> None:
    wandb.init(project="transformer-memorization-hashed", config=cfg)

    # instantiate GPT-like decoder model

    model = Transformer(
        cfg
    )

    model = AutoregressiveWrapper(model)
    model.cuda()

    train_dataset = RandomBytesDataset(seqlen=cfg.max_seq_len + 1, length=cfg.dataset_size, inline_meta=cfg.inline_meta)
    val_dataset   = RandomBytesDataset(seqlen=cfg.max_seq_len + 1, length=cfg.dataset_size, inline_meta=cfg.inline_meta)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = cfg.batch_size, shuffle=cfg.shuffle))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = cfg.batch_size, shuffle=cfg.shuffle))

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # training

    for i in tqdm.tqdm(range(cfg.num_batches), mininterval=10., desc='training'):
        start_time = time.time()
        model.train()

        for __ in range(cfg.gradient_accumulate_every):
            loss = model(next(train_loader))
            loss.backward()

        end_time = time.time()
        print(f'training loss: {loss.item()}')
        train_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()

        train_dataset.step()
        val_dataset.step()


        if i % cfg.validate_every == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')
                val_loss = loss.item()

        if i % cfg.generate_every == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp, cfg.generate_legnth)
            output_str = decode_tokens(sample)
            print(output_str)
        
        logs = {}
        
        logs = {
          **logs,
          'iter': i,
          'step_time': end_time - start_time,
          'train_loss': train_loss,
          'val_loss': val_loss,
        }
        
        wandb.log(logs)
      
    wandb.finish()

if __name__ == '__main__':
    train()
