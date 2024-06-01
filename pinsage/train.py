import os
import pickle

import dgl

import evaluation
import layers
import numpy as np
import sampler as sampler_module
import torch
import torch.nn as nn
import torchtext
import tqdm
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from model import *
import json

#Load dataset
dataset_path = "processed_pinsage_Amazon_electronics.pkl"
with open(dataset_path, "rb") as f:
	dataset = pickle.load(f)
	
# Load config
config_path = "./config/pinsage-params.json"
with open(config_path, "rb") as f:
    model_config = json.load(f)

class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

cfg = DictToObject(model_config)

model, h_item, loss_list, hit_list = train(dataset, cfg)
