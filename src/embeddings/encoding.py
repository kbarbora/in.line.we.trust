import logging

import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
import utility.config as configs
import os
from networkx.drawing import nx_agraph
import pygraphviz as pgv
from sklearn.feature_extraction.text import TfidfVectorizer

PATHS = configs.Paths()
EMBED = configs.Embed()
PROJ_ROOT = os.path.abspath('.')

def _tokenize_code(dataset_code: pd.Series) -> pd.Series:
    _pattern = re.compile(EMBED.tokenize_regex)
    # tokenizer = AutoTokenizer.from_pretrained("maximus12793/CodeBERTa-small-v1-finetuned-cpp")
    # _tokens = pd.Series()
    # HEADER:
    # PROJECT:str,COMMIT:str, FILENAME:str,INSTANCES:int,CODE LINES:list,LINES TOKENS:list,ENCODED LINES:list
    _token_series = pd.Series()
    _tokenizer = EMBED.tokenize_regex
    # dataset_code is a pandas series containing a list with the lines of the source code
    for i, function in dataset_code.items():
        _tokenized = []
        for _line in function:
            _tokens = re.findall(_pattern, _line)
            if _tokens == [""]:
                logging.error(f"Empty tokens after running regex for line: {_line}")
            # Delete all empty tokens found
            [_tokens.pop(ndx) if token == " " else token for ndx, token in enumerate(_tokens)]
            _tokenized.append(_tokens)
        _token_series.append(_tokenized)

    return _token_series

def _tokenize_nodes(nodes_att: pd.Series) -> pd.Series:
    tokenized = {}
    return

def _tokenize_edges(edges_att: pd.Series) -> pd.Series:
    tokenized = {}
    return

def tokenize_graph(graphs: pd.Series) -> pd.Series:
    _tokenize_code(graphs)
    _tokenize_nodes(graphs)
    _tokenize_edges(graphs)
    return