from pathlib import Path
from abc import *
from config import RAW_DATASET_ROOT_FOLDER


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def maybe_download_raw_dataset(self):
        pass

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())