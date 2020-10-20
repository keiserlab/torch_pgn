from abc import ABC, abstractmethod

class PGDataset(ABC):
    """Generic class for loading datasets. It has all of the methods required to write the dataset in order to be
    compatible with the rest of the processing pipeline."""
    def __init__(self, args):
        self.num_workers = args.num_workers
        self.data_path = args.data_path
        self.graphs = []

    @abstractmethod
    def process_raw_data(self):
        pass

    def write_graphs(self):
