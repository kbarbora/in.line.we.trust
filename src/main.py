
import logging
import os
import pandas as pd
from os.path import join as pathjoin
from argparse import ArgumentParser
import utility.config as config
import preprocess.scrapper as scrapper
import preprocess.data_processing as data_processing


PATHS = config.Paths()
PROJ_ROOT = os.path.abspath('.')  # path of the root project
def preprocess(project: str):
    """

    :param project:
    :return:
    """
    # ---------- Check for pre-process dataset, if not found run the scrapper function ---------------
    dataset_path = pathjoin(PATHS.raw, project, PATHS.vuln_source_code)
    if not os.path.exists(dataset_path):
        raw = pd.read_json(PATHS.raw, project, PATHS.vulnerable_lines_json)  # read original 'CarrotBlender' dataset published by the authors
        scrapper.scrape_dataset(project)
    dataset = pd.read_pickle(dataset_path)
    logging.info(f"Dataset memory information: \n{dataset.info(memory_usage='deep')}")

    # ---------- Check for cpg files, if not found run joern-parse to create them ---------------
    data_processing.create_cpg(project)
    # ----------- Check for graphs, if not found run joern-export to create them ----------------
    graphs_paths = data_processing.extract_graph(project)
    # ----------- Load graphs
    graphs_data = data_processing.load_graphs(project)
    data_processing.append_graph_to_dataset(dataset, graphs_data)



def main():
    """
    main function that executes tasks based on command-line options
    """
    parser = ArgumentParser()
    # parser.add_argument('-p', '--prepare', help='Prepare task', required=False)
    parser.add_argument('project', default="all", help="The Project to be process"
                        " (options are ffmpeg, imagemagick, php, openssl, linux or all). Default is 'all'.")
    parser.add_argument('-pP', '--preprocess', action='store_true')
    parser.add_argument('-e', '--embed', action='store_true')
    parser.add_argument('-jG', '--joint_graph', action='store_true')
    parser.add_argument('-m', '--max-memory', action='store', dest='max_java_mem', default=8,
                        help="Max memory that java is going to be allowed to use.")
    args = parser.parse_args()
    if not args.project:
        logging.error("[Error] - Choose ONE between ffmpeg, imagemagick, php, openssl or linux datasets.")
    if int(args.max_java_mem) > 0:
        os.environ["JAVA_OPTS"] = f"-Xmx{args.max_java_mem}G"
    if args.preprocess:
        proj = args.project
        if not (('ffmpeg' == proj) or ('php' == proj) or ('openssl' == proj) or ('linux' == proj)
                or ('imagemagick' == proj) or ('all' == proj)):
            logging.error("[Error] - Project not identified. Exit")
            exit()
        preprocess(args.project)


if __name__ == "__main__":
    logging.basicConfig(filename="main.log", level=logging.INFO)
    main()
    logging.info("Finish main function. Exit")
