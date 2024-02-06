"""
# Author: Yinghao Li
# Modified: September 13th, 2023
# ---------------------------------------
# Description: dataset loading and processing
"""

import os
import json
import logging
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer
from transformers import BertTokenizer

from src.args import Config
from src.utils.data import span_to_label, span_list_to_dict
from .batch import pack_instances

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

MASKED_LB_ID = -100


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text: Optional[List[List[str]]] = None, lbs: Optional[List[List[str]]] = None):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._token_ids = None
        self._attn_masks = None
        self._bert_lbs = None

        self._partition = None

        self.data_instances = None

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        #print('idx',idx)
        #print((self.data_instances))
        return self.data_instances[idx]

    def prepare(self, config: Config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """
        assert partition in ["train", "valid", "test"], ValueError(
            f"Argument `partition` should be one of 'train', 'valid' or 'test'!"
        )
        self._partition = partition

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))
        logger.info(f"Loading data file: {file_path}")
        self._text, self._lbs = load_data_from_json(file_path)
    
        logger.info("Encoding sequences...")
        self.encode(config.bert_model_name_or_path, {lb: idx for idx, lb in enumerate(config.bio_label_types)})

        logger.info(f"Data loaded.")

        #print('self._token_ids',self._token_ids)

        self.data_instances = pack_instances(
            bert_tk_ids=self._token_ids,
            bert_attn_masks=self._attn_masks,
            bert_lbs=self._bert_lbs,
        )
        return self

    def encode(self, tokenizer_name: str, lb2idx: dict):
        
        """
        Build BERT token masks as model input

        Parameters
        ----------
        tokenizer_name: str
            the name of the assigned Huggingface tokenizer
        lb2idx: dict
            a dictionary that maps the str labels to indices

        Returns
        -------
        self
        """

        def align_tokens_and_labels(tokens, labels):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tokenized_texts = []
            new_labels = []

            for word, label in zip(tokens, labels):
                # Tokenize the word and count # of subwords the word is split into
                tokenized_word = tokenizer.tokenize(word)
                n_subwords = len(tokenized_word)

                # Add the tokenized word to the final tokenized word list
                tokenized_texts.extend(tokenized_word)

                # Add the same label to the new list of labels `n_subwords` times
                new_labels.extend([label] * n_subwords)

            return tokenized_texts, new_labels

        def update_labels(tokens, aligned_labels):
            updated_labels = []
            for i, token in enumerate(tokens):
                if token.startswith('##'):
                    updated_labels.append(-100)
                    #updated_labels.append(0)
                else:
                    updated_labels.append(aligned_labels[i])
            return updated_labels

        print(tokenizer_name)
        #print('lb2idx',lb2idx)
        #print('_text',self._text)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        tokenized_text = tokenizer(self._text, add_special_tokens=True, is_split_into_words=True)

        self._token_ids = tokenized_text.input_ids
        self._attn_masks = tokenized_text.attention_mask

        aligned_tokens, aligned_labels = align_tokens_and_labels(self._text[1], self._lbs[1])        

        # Update label sequence to match the BERT tokenization.
        # The labels that are not involved in the loss calculation should be masked out by `MASKED_LB_ID`.
        # Hint: labels corresponding to [CLS], [SEP], and non-first subword tokens should be masked out.
        # You should store the updated label sequence in the `bert_lbs_list` variable.
        
        # --- TODO: start of your code ---
        # self._lbs is the list of original labels for the text
        bert_lbs_list = []
        i=0
        for t_id in self._text:
            bert_lbs = []
            bert_lbs.append(MASKED_LB_ID)
            aligned_tokens, aligned_labels = align_tokens_and_labels(self._text[i], self._lbs[i])
            new_aligned_labels = update_labels(aligned_tokens, aligned_labels)
        
            bert_lbs.extend(new_aligned_labels)
            bert_lbs.append(MASKED_LB_ID)
            bert_lbs_list.append(bert_lbs)

            '''
            print('********')
            print('self._text[i]',self._text[i])
            print('aligned_tokens',aligned_tokens)
            print('aligned_labels',aligned_labels)
            print('blue_label',bert_lbs)
            '''

            i+=1
        # Check if lengths match
        assert len(self._token_ids) == len(bert_lbs_list)
        # --- TODO: end of your code ---

        for tks, lbs in zip(self._token_ids, bert_lbs_list):
            #print('tks',tks)
            #print('lbs',lbs)
            assert len(tks) == len(lbs), ValueError(
                f"Length of token ids ({len(tks)}) and labels ({len(lbs)}) mismatch!"
            )

        self._bert_lbs =  bert_lbs_list

        return self


def load_data_from_json(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: str
        file directory

    """
    with open(file_dir, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    tk_seqs = list()
    lbs_list = list()

    for inst in data_list:
        # get tokens
        tk_seqs.append(inst["text"])

        # get true labels
        lbs = span_to_label(span_list_to_dict(inst["label"]), inst["text"])
        lbs_list.append(lbs)

    return tk_seqs, lbs_list
