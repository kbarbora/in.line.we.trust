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
    """
    Tokenize all the code for the current project, not recommended to use ALL projects since the vocabulary size will
    be affected
    :param dataset_code: A pandas series object having as elements lists of the code of every function
    (list contain each line of code in string format)
    :return: A pandas series of the code tokenized
    """
    _pattern = re.compile(EMBED.tokenize_regex)
    # tokenizer = AutoTokenizer.from_pretrained("maximus12793/CodeBERTa-small-v1-finetuned-cpp")
    # _tokens = pd.Series()
    # HEADER:
    # PROJECT:str,COMMIT:str, FILENAME:str,INSTANCES:int,CODE LINES:list,LINES TOKENS:list,ENCODED LINES:list
    _token_series = pd.Series()
    _tokenizer = EMBED.tokenize_regex
    # @TODO: Add signature at the beginning of the code before tokenization

    # dataset_code is a pandas series containing a list with the lines of the source code
    for i, function in dataset_code.items():
        _tokenized = []
        for _line in function:
            _tokens_line = re.findall(_pattern, _line)
            if _tokens_line == [""]:
                logging.error(f"Empty tokens after running regex for line: {_line}")
            # _tokens_line.append("<NL>") # Need to further test if NL token is needed
            # Delete all empty tokens found
            [_tokens_line.pop(ndx) if token == " " else token for ndx, token in enumerate(_tokens_line)]
            _tokenized.append(_tokens_line)
        _token_series.append(_tokenized)
    return _token_series

def vectorize_code(token_series: pd.Series):
    """
    Vectorize code passed as argument of the function. The argument represents the code already tokenized.
    This function uses TF-IDF vectorizer (not replacing variables names) limiting to the top 3000 words in the
    vocabulary.
    :param token_series: A pandas series representing the tokenized code to encode and vectorize
    :return: A pandas series object representing the vectorized code
    """
    # ignore tokenizer and preprocessor with dummy function since input is already tokenized
    tfidf = TfidfVectorizer(analyzer='word', max_features=3000,
                            tokenizer=dummy_function, preprocessor=dummy_function, token_pattern=None)
    vectors = tfidf.transform([token_series])  # vectors is a sparse matrix
    code_vectors = []
    for vector in range(vectors.shape[0]):  # iterate over each row
        array = vector.toarray()
        array_clean = array[array != 0]  # delete zeroes entries
    code_vectors.append(array_clean)
    return pd.Series(code_vectors)


def _vectorize_nodes(nodes_att: pd.Series) -> pd.Series:
    """
    Vectorize the nodes attributes (not including the code). Currently not used.
    :param nodes_att:
    :return:
    """
    vectorize = {}

    return

def _tokenize_edges(edges_att: pd.Series) -> pd.Series:
    tokenized = {}
    return

def tokenize_graph(graphs: pd.Series) -> pd.Series:
    _tokenize_code(graphs)
    _vectorize_nodes(graphs)
    _tokenize_edges(graphs)
    return

def dummy_function(_doc):
    return _doc