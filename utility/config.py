import json
import torch

class Config(object):
    def __init__(self, config, file_path="configs.json"):
        with open(file_path) as config_file:
            self._config = json.load(config_file)
            self._config = self._config.get(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_property(self, property_name):
        return self._config.get(property_name)

    def get_device(self):
        return self.device

    def all(self):
        return self._config

class Preprocess(Config):
    def __init__(self):
        super().__init__('create')

    @property
    def filter_column_value(self):
        return self.get_property('filter_project')

    @property
    def slice_size(self):
        return self.get_property('slice_size')

    @property
    def joern_cli_dir(self):
        return self.get_property('joern_cli_dir')


# class Data(Config):
#     def __init__(self, config):
#         super().__init__(config)
#
#     @property
#     def projects(self):
#         return {"ffmpeg": "FFmpeg/FFmpeg", "imagemagick": "ImageMagick/ImageMagick", "linux": "torvalds/linux",
#                 "openssl": "openssl/openssl", "php": "php/php-src"}

class Paths(Config):
    def __init__(self):
        super().__init__('paths')

    @property
    def graph(self):
        return self.get_property('graph')

    @property
    def cpg(self):
        return self.get_property('cpg')

    @property
    def cpg_pickle(self):
        return self.get_property('cpg_pickle')

    @property
    def struct(self):
        return self.get_property('struct')

    @property
    def raw(self):
        return self.get_property('raw')

    @property
    def input(self):
        return self.get_property('input')

    @property
    def model(self):
        return self.get_property('model')

    @property
    def tokens(self):
        return self.get_property('tokens')

    # As far as I could tell, this should be w2v
    # @property
    # def w1v(self):
    #     return self.get_property('w1v')

    @property
    def w2v(self):
        return self.get_property('w2v')

    @property
    def stdout(self):
        return self.get_property('stdout')

    @property
    def temp(self):
        return self.get_property('temp')

    @property
    def joern(self):
        return self.get_property('joern')

    @property
    def vuln_source_code(self):
        return self.get_property('vuln_source_code')

    @property
    def vulnerable_lines_json(self):
        return self.get_property('vulnerable_lines')


class Data(Config):
    def __init__(self):
        super().__init__('data')

    @property
    def tokens(self):
        return self.get_property('tokens')

    @property
    def w2v(self):
        return self.get_property('w2v')

    @property
    def raw(self):
        return self.get_property('raw')

    @property
    def projects(self):
        return self.get_property('projects')

    @property
    def graph_structures(self):
        return self.get_property('graph_structures')

class Embed(Config):
    def __init__(self):
        super().__init__('embed')

    @property
    def nodes_dim(self):
        return self.get_property('nodes_dim')

    @property
    def tokenize_regex(self):
        return r"""(\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""

    @property
    def w2v_args(self):
        return self.get_property('word2vec_args')

    @property
    def edge_type(self):
        return self.get_property('edge_type')


class Process(Config):
    def __init__(self):
        super().__init__('process')

    @property
    def epochs(self):
        return self.get_property('epochs')

    @property
    def patience(self):
        return self.get_property('patience')

    @property
    def batch_size(self):
        return self.get_property('batch_size')

    @property
    def dataset_ratio(self):
        return self.get_property('dataset_ratio')

    @property
    def shuffle(self):
        return self.get_property('shuffle')


class Devign(Config):
    def __init__(self):
        super().__init__('devign')

    @property
    def learning_rate(self):
        return self.get_property('learning_rate')

    @property
    def weight_decay(self):
        return self.get_property('weight_decay')

    @property
    def loss_lambda(self):
        return self.get_property('loss_lambda')

    @property
    def model(self):
        return self.get_property('model')
