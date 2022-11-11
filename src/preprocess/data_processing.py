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
import src.data as data
from data import datamanager as data

PATHS = configs.Paths()


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


def create_cpg(project: str, dataset: pd.DataFrame, javaheap: int):
    """
    Get the vulnerable functions and create the cpg files to later extract the functions.
    The main problem is that there are different definitions of a function (with same name)
     across commits. If parsed it through Joern, it will get the latest read only.
    """
    if "JAVA_OPTS" not in os.environ:
        logging.info(f"Setting Java HEAP to a max of {javaheap}GB")
        os.environ["JAVA_OPTS"] = f"-Xms128M -Xmx{javaheap}G"
    else:
        logging.info(f"Java HEAP environment variable already set by external process to {javaheap}GB. Continue.")

    _sourcefiles = pathjoin(PATHS.raw, "functions")
    _outfile = pathjoin(PATHS.cpg, f"{project}.bin")
    if os.path.exists(_outfile):
        logging.warning(f"File {_outfile} exists. Skipping.")
    else:
        logging.info(f"Creating CPG for source project {project}")
        subprocess.run([pathjoin(".", PATHS.joern, "joern-parse"), _sourcefiles, "--output", _outfile],
                       stdout=subprocess.PIPE, text=True, check=True)
        logging.info(f"CPG file created: {_outfile}")
    return _outfile


def create_struct(cpg_files: list, joint_graph=False, joern_path="joern-cli/") -> dict:
    output = {}
    run_structs = False

    for cpg in cpg_files:
        cpg_file = cpg.split('/')[-1]
        project = cpg.split('/')[-2]
        dir = pathjoin(PATHS.struct, project, cpg_file)
        if not os.path.exists(dir):
            run_structs = True
            os.mkdir(dir)
        output[dir] = {}
        if not joint_graph:
            structs = ["all"]
        else:
            structs = STRUCTS
        for s in structs:
            out_dir = pathjoin(dir, s)
            # timer = time.time()
            input_dir = pathjoin(PATHS.raw, project, "slices", cpg_file[:-4])
            if run_structs:
                subprocess.run([f"./{joern_path}joern-export", "--repr", s, "--out", out_dir,
                                pathjoin(PATHS.cpg, project, cpg_file)], check=True)
            output_files = sorted(os.listdir(out_dir))
            logging.info(f"[Info] - Extracted {s} from dataset {cpg_file} producing {len(output_files)} files. ")
            print(f"[Info] - Extracted {s} from dataset {cpg_file} producing {len(output_files)} files. ")
            if not os.path.exists(pathjoin(out_dir, "clean")):
                output[dir][s] = graph_generator.clean_graph_structs(s, input_dir, out_dir)
                clean_file = open(pathjoin(out_dir, "clean"), 'w+')
                clean_file.close()
            else:
                output[dir][s] = os.listdir(pathjoin(dir, s))

    # at this point all structs needed for all dataset should be created.
    return output