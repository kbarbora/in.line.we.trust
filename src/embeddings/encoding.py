import logging

import networkx
import pandas as pd
import re
# from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    Tokenize all the code for the current project, not recommended to use for ALL projects since the vocabulary size
     will be affected and can explode.
    :param dataset_code: A pandas series object having as elements lists of the code of every function
    (list contain each line of code in string format)
    :return: A pandas series of the code tokenized
    """

    # HEADER:
    # PROJECT:str,COMMIT:str, FILENAME:str,INSTANCES:int,CODE LINES:list,LINES TOKENS:list,ENCODED LINES:list
    _pattern = re.compile(EMBED.tokenize_regex)
    _token_series = []
    _tokenizer = EMBED.tokenize_regex
    # @TODO: Add signature at the beginning of the code before tokenization

    # dataset_code is a pandas series containing lists with the lines of the source code
    for i, function in dataset_code['sourcecode'].items():
        _tokenized = []
        for _line in function:
            _line = re.sub(r'\s+', ' ', _line)  # delete word/line delimiter chars and replace them for a single space
            _tokens_line = re.findall(_pattern, _line)
            if _tokens_line == [""]:
                logging.error(f"Empty tokens after running regex for line: {_line}")
            # _tokens_line.append("<NL>") # Need to further test if NL token is needed
            # Delete all empty tokens found
            [_tokens_line.pop(ndx) if token == " " else token for ndx, token in enumerate(_tokens_line)]
            if len(_tokens_line) == 0:
                continue
            _tokenized.append(_tokens_line)
        # _tokenized = pd.Series(_tokenized)
        print(_tokenized)
        _token_series.append(_tokenized)
    return pd.Series(_token_series)


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
    tfidf.fit(token_series)  # learn vocabulary from all the existing tokens
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

    return pd.Series(vectorize)


def _tokenize_att(graphs: list) -> list:
    for graph in graphs:
        # NODES
        _delete_node_attribute(graph, EMBED.node_attr_to_delete)
        for node_id, node_attr in graph.nodes(data=True):
            # node_id: str, node_attr: dict
            _attr_to_check = "METHOD_FULL_NAME"
            if _attr_to_check in node_attr:
                # attr: METHOD_FULL_NAME, reason: simplify
                # sample METHOD_FULL_NAME="<operator>.indirectFieldAccess"
                _tmp = node_attr[_attr_to_check].replace("<operator>.", "")
                graph[node_id][_attr_to_check] = _tmp
        _encode_node_attribute(graph, EMBED.node_encoding)
        # EDGES
        _delete_edge_attribute(graph, EMBED.edge_attr_to_delete)
        _encode_edge_attribute(graph, EMBED.edge_encoding)
    return graphs


# def _drop_attribute_keys(graph: networkx.DiGraph, line_no_first_att: bool = True) -> networkx.DiGraph:
#     # NODES
#     for node_id, node_attr in graph.nodes(data=True):
#
#
#     # EDGES
#     for edge_from, edge_to, edge_attr in graph.edges(data=True):
#
#     return graph


def _delete_node_attribute(graph: networkx.DiGraph, attributes_to_delete: list):
    # stats = {}
    for att in attributes_to_delete:
        for node_id, node_attr in graph.nodes(data=True):
            if att in node_attr:
                del graph.nodes[node_id][att]
                # if att in stats.keys():
                #     stats[att] += 1
                # else:
                #     stats[att] = 1
    return


def _delete_edge_attribute(graph: networkx.DiGraph, attributes_to_delete: list):
    for att in attributes_to_delete:
        for edge_from, edge_to, edge_attr in graph.edges(data=True):
            edge_attr.pop(att, None)
    return


def _encode_node_attribute(graph: networkx.DiGraph, attributes_to_encode: dict):
    """
    Encode the node attributes of the parameter graph by using the mapping contained in the second param
    attributes_to_encode. Since graph arg is passed by reference, no return value is needed.
    :param graph: A networkx DiGraph object representing the graph to encode the node attributes
    :param attributes_to_encode:  A dict object containing all the attributes names as key and as value, the mapping
                                  from value to encoded numerical value
    """
    for att, mapping in attributes_to_encode.items():
        for node_id, node_attr in graph.nodes(data=True):
            if att in node_attr.keys():
                tmp = node_attr[att]
                if att == 'TYPE_FULL_NAME':
                    if tmp not in mapping.values():
                        logging.info(f"Custom graph attributed used instead of {tmp} for Category {att}")
                        tmp = "CUSTOM"
                node_attr[att] = mapping[tmp]
            else:   # att NOT found in graph, fill it with -1 (empty) since all graphs have to have the same atts
                node_attr[att] = -1
    return


def _encode_edge_attribute(graph: networkx.DiGraph, attributes_to_encode: dict):
    """
    Encode the edge attributes of the parameter graph by using the mapping contained in the second param
    attributes_to_encode. Since graph arg is passed by reference, no return value is needed.
        :param graph: A networkx DiGraph object representing the graph to encode the node attributes
        :param attributes_to_encode:  A dict object containing all the attributes names as key and as value, the mapping
                                      from value to encoded numerical value

    """
    for att, mapping in attributes_to_encode.items():
        for edge_from, edge_to, edge_attr in graph.edges(data=True):
            if att in edge_attr.keys():
                edge_attr[att] = mapping[edge_attr[att]]


def tokenize_graph(graphs: pd.Series) -> pd.Series:

    code_tokens = _tokenize_code(graphs)
    code_vectors = vectorize_code(code_tokens)
    # _tokenize_edges(graphs)
    return code_vectors

def dummy_function(_doc):
    return _doc