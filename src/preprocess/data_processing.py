import errno
import glob
import json
import logging
import math
import pickle
import re
import shutil
import subprocess
import os.path
import os
import sys
from os.path import join as pathjoin
import logging
import pandas as pd
import networkx as nx
import utility.config as configs
from torch_geometric.data import Data as GraphData
from networkx.drawing import nx_agraph
import pygraphviz as pgv

PATHS = configs.Paths()
PROJ_ROOT = os.path.abspath('.')  # path of the root project


def _create_slices(project: str, project_dataset: pd.DataFrame) -> int:
    """
    Since there are multiple vulnerabilities through
    """
    slices = {}
    max_len = 1
    # Calculate the slices
    for i in project_dataset.iterrows():
        commit = i[0]
        this = i[1]  # 0 correspond to index, 1 to element data
        func = this.function_name
        if func in slices.keys():
            if commit in slices[func]:
                logging.warning(f"Commit '{commit}' already present in function '{func}'")
                continue
            slices[func].append(commit)
            this_len = len(slices[func])
            if this_len > max_len:
                max_len = this_len
        else:
            slices[func] = [commit, ]
    #  Copy source code files to disk
    for i in range(max_len):
        _tmp = []
        for f in slices.keys():
            _this = slices[f]
            if len(_this) == 0:
                logging.error(f"Slice with key {_this} did not contain any record. Skip.")
                continue
            _tmp_name = f"{f}--{_this.pop()}.c"
            _tmp.append(pathjoin(PATHS.raw, project, "commits", _tmp_name))
        _slices_path = pathjoin(PATHS.raw, project, "slices", str(i))
        if not os.path.exists(_slices_path):
            os.makedirs(_slices_path)
            for t in _tmp:
                os.symlink(pathjoin(os.getcwd(), t), pathjoin(os.getcwd(), _slices_path, t.split('/')[-1]))
                # shutil.copy(t, f"{_slices_path}")  # @TODO use symlinks instead of coping the files
    return max_len


def create_cpg(project: str, javaheap: int = 4):
    """
    Get the vulnerable functions and create the cpg files to later extract the functions.
    The main problem is that there are different definitions of a function (with same name)
     across commits. If parsed it through Joern, it will get the latest read only.
    """
    if "JAVA_OPTS" not in os.environ:
        logging.info(f"Setting Java HEAP to a max of {javaheap}GB")
        os.environ["JAVA_OPTS"] = f"-Xms128M -Xmx{javaheap}G"
    else:
        logging.info(f"Java HEAP environment variable already set to {javaheap}GB. Continue.")

    _sourcefiles = pathjoin(PATHS.raw, project, "functions")
    _outfile = pathjoin(PATHS.cpg, f"{project}.bin")
    if os.path.exists(_outfile):
        logging.warning(f"File {_outfile} exists. Skipping.")
    else:
        logging.info(f"Creating CPG for source project {project}")
        subprocess.run([pathjoin(".", PATHS.joern, "joern-parse"), _sourcefiles, "--output", _outfile],
                       stdout=subprocess.PIPE, text=True, check=True)
        logging.info(f"CPG file created: {_outfile}")
    return _outfile


def extract_graph(project: str) -> str:
    output_path = pathjoin(PATHS.graph, project)
    cpg_path = pathjoin(PATHS.cpg, project+'.bin')
    if os.path.exists(output_path):
        logging.warning(f"GRAPH output path exists {output_path}. Return existing files.")
        return os.listdir(output_path)
    temp_path = pathjoin(PATHS.temp, project)
    subprocess.run([f"./{pathjoin(PATHS.joern, 'joern-export')}", "--repr", "cpg", "--out", temp_path,
                    cpg_path], check=True)
    # @TODO rename output to remove extension
    # Clean output files structure
    temp_output_files = pathjoin(temp_path, "_root_") + PROJ_ROOT + '/' + pathjoin(PATHS.raw, project, "functions")
    os.rename(temp_output_files, output_path)
    shutil.rmtree(temp_path)
    for dir, _, graph in os.walk(output_path):
        if dir == output_path:
            continue
        if len(graph) == 1:
            logging.warning(f"Graph output did not produced function. Only {graph[0]}.")
            shutil.rmtree(dir)
            continue
        curr = pathjoin(dir, graph[-1])
        file_commit = dir.split(os.path.sep)[-1][0:-2]
        os.rename(curr, pathjoin(output_path, file_commit+'.dot'))
        shutil.rmtree(dir)
    logging.info(f"[Info] - Extracted {len(os.listdir(output_path))}graphs from dataset {cpg_path}.")
    return output_path


def load_graphs(project: str) -> dict:
    input_graphs = pathjoin(PATHS.graph, project)
    output_graphs = {}
    _, _, graphs = next(os.walk(input_graphs))
    for graph in graphs:
        graph_path = pathjoin(input_graphs, graph)
        graph = graph.split('.')[0]  # remove extension
        g = nx_agraph.from_agraph(pgv.AGraph(graph_path))
        graphdata = GraphData()
        for node, node_attr in g.nodes(data=True):
            graphdata[node] = node_attr
        graphdata.edge_index = list(g.nodes())

        # g = nx.Graph(nx.nx_pydot.read_dot(graph_path))  # uses pydot module
        output_graphs[graph] = g



    return output_graphs

def append_graph_to_dataset(dataset: pd.DataFrame, output_graphs: dict) -> pd.DataFrame:
    empty_list = [""]*len(output_graphs)
    dataset['graph'] = empty_list
    for _, data in dataset.iterrows():
        _function_name = data['function_name']
        _commit = data['commit']
        _full_name = _commit + '--' + _function_name
        if output_graphs[_full_name] == "" or isinstance(output_graphs[_full_name], GraphData):
            continue
        # update dataset in place
        dataset.loc[(dataset.function_name == _function_name) & (dataset.commit == _commit),
                    ['graph']] = output_graphs[_full_name]
    return dataset

