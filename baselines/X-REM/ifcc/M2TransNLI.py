import gzip
import json
import os
import site
import sys
import time
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
import torch
from clinicgen.models.bertnli import BERTNLI
from clinicgen.text.sentsplit import get_sentsplitter
from clinicgen.text.tokenizer import get_tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import DataParallel
from torch.nn.functional import softmax
from transformers import AutoTokenizer


# Adapted ifcc codebase
class M2TransNLI:
    CONFIG_FILENAME = "config.json"
    DEFAULT_BERT_TYPE = "bert"
    DEFAULT_NAME = "bert-base-uncased"
    DEFAULT_STATES = "model_mednli_13k"
    RADNLI_STATES = "model_medrad_19k"
    STATES_FILENAME = "model.dict.gz"

    def __init__(
        self,
        model,
        neutral_score=(1.0 / 3),
        batch=16,
        nthreads=2,
        pin_memory=False,
        bert_score=None,
        sentsplitter="none",
        cache=None,
        verbose=False,
    ):
        self.neutral_score = neutral_score
        self.batch = batch
        self.nthreads = nthreads
        self.pin_memory = pin_memory
        self.bert_score_model = None
        self.cache = None
        self.verbose = verbose
        self.sentsplitter = get_sentsplitter(sentsplitter, linebreak=False)
        self.tokenizer = get_tokenizer("nltk")
        self.model = model
        self.model.eval()
        self.gpu = False

    @classmethod
    def load_model(cls, states=None):
        resource_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "resources"
        )
        if states is None:
            name = cls.DEFAULT_NAME
            states = os.path.join(resource_dir, cls.DEFAULT_STATES, cls.STATES_FILENAME)
            bert_type = cls.DEFAULT_BERT_TYPE
        else:
            config_path = os.path.join(states, cls.CONFIG_FILENAME)
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            bert_type = config["bert_type"]
            if bert_type == "bert":
                name = "bert-base-uncased"
            elif bert_type == "distilbert":
                name = "distilbert-base-uncased"
            else:
                raise ValueError("Unknown BERT type {0}".format(bert_type))
            states = os.path.join(states, cls.STATES_FILENAME)
        bertnli = BERTNLI(
            name, bert_type=bert_type, length=384, force_lowercase=True, device="cpu"
        )
        with gzip.open(states, "rb") as f:
            states_dict = torch.load(f, map_location=torch.device("cpu"))
        bertnli.load_state_dict(states_dict)
        return bertnli

    def predict(self, sent1s, sent2s):
        batches, buf1, buf2 = [], [], []
        for sent1, sent2 in zip(sent1s, sent2s):
            buf1.append(sent1)
            buf2.append(sent2)
            if len(buf1) >= self.batch:
                batches.append((buf1, buf2))
                buf1, buf2 = [], []
        if len(buf1) > 0:
            batches.append((buf1, buf2))

        probs, preds = [], []
        with torch.no_grad():
            for b1, b2 in batches:
                out = self.model(b1, b2)
                out = softmax(out, dim=-1).detach().cpu()
                _, idxs = out.max(dim=-1)
                for i, idx in enumerate(idxs):
                    idx = int(idx)
                    probs.append(
                        {
                            "entailment": float(out[i][BERTNLI.LABEL_ENTAILMENT]),
                            "neutral": float(out[i][BERTNLI.LABEL_NEUTRAL]),
                            "contradiction": float(out[i][BERTNLI.LABEL_CONTRADICTION]),
                        }
                    )
                    if idx == BERTNLI.LABEL_ENTAILMENT:
                        preds.append("entailment")
                    elif idx == BERTNLI.LABEL_NEUTRAL:
                        preds.append("neutral")
                    elif idx == BERTNLI.LABEL_CONTRADICTION:
                        preds.append("contradiction")
                    else:
                        raise ValueError("Unknown label index {0}".format(idx))
        return probs, preds
