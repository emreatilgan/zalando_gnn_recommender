from torch_geometric.data import Dataset, Data
import torch
from torch_geometric.loader import NeighborLoader
import pandas as pd
import numpy as np
from .graph_builder import ZalandoGraphBuilder

class ZalandoDataset(Dataset):
    def __init__(self, root, config, transform=None, pre_transform=None):
        self.config = config
        super().__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return ['interactions.csv', 'user_features.csv', 'content_features.csv']
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        # Download from your data source
        pass
    
    def process(self):
        # Read raw data
        interactions_df = pd.read_csv(self.raw_paths[0])
        user_features_df = pd.read_csv(self.raw_paths[1])
        content_features_df = pd.read_csv(self.raw_paths[2])
        
        # Build graph
        graph_builder = ZalandoGraphBuilder(self.config)
        data = graph_builder.build_graph(
            interactions_df,
            user_features_df,
            content_features_df
        )
        
        # Save processed data
        torch.save(data, self.processed_paths[0])
        
    def len(self):
        return 1
        
    def get(self, idx):
        data = torch.load(self.processed_paths[0])
        return data

def create_dataloader(dataset, config, mode='train'):
    """Create dataloader with neighborhood sampling."""
    return NeighborLoader(
        dataset[0],
        num_neighbors=config.NUM_NEIGHBORS,
        batch_size=config.BATCH_SIZE,
        input_nodes=('user', 'content'),
        shuffle=(mode == 'train')
    )