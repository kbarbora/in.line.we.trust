import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForMaskedLM
import utility.config as configs
import os
from networkx.drawing import nx_agraph
import pygraphviz as pgv

PATHS = configs.Paths()
EMBED = configs.Embed()
PROJ_ROOT = os.path.abspath('.')

def _tokenize_code(nodes_code: pd.Series) -> pd.Series:
    _pattern = EMBED.tokenize_regex
    tokenizer = AutoTokenizer.from_pretrained("maximus12793/CodeBERTa-small-v1-finetuned-cpp")
    _tokens = pd.Series()
    for i, _code in nodes_code.items():


    return

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