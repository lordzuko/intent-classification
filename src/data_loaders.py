from __future__ import annotations

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

from config import LABEL_COLUMNS


class IntentClassificationDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: BertTokenizerFast,
        max_token_len: int = 100,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row.text
        lang = data_row.lang
        labels = data_row[LABEL_COLUMNS]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return dict(
            text=text,
            lang=lang,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels),
        )


class IntentClassificationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: BertTokenizerFast,
        batch_size=8,
        max_token_len=100,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        self.train_dataset = IntentClassificationDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len,
        )

        self.val_dataset = IntentClassificationDataset(
            self.valid_df,
            self.tokenizer,
            self.max_token_len,
        )

        self.test_dataset = IntentClassificationDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2)
