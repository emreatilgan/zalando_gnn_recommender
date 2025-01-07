import torch
from torch_geometric.data import Data
import numpy as np
from collections import defaultdict

class ZalandoGraphBuilder:
    def __init__(self, config):
        self.config = config
        self.user_mapping = {}
        self.content_mapping = {}
        self.user_features = {}
        self.content_features = {}
        
    def _create_mappings(self, interactions_df):
        """Create mappings for user and content IDs to consecutive indices."""
        unique_users = interactions_df['user_id'].unique()
        unique_content = interactions_df['content_id'].unique()
        
        self.user_mapping = {uid: idx for idx, uid in enumerate(unique_users)}
        self.content_mapping = {cid: idx + len(self.user_mapping) for idx, cid in enumerate(unique_content)}
        
    def _create_edge_index(self, interactions_df, interaction_type='view'):
        """Create edge index for PyTorch Geometric from interactions."""
        edges = []
        for _, row in interactions_df.iterrows():
            user_idx = self.user_mapping[row['user_id']]
            content_idx = self.content_mapping[row['content_id']]
            if interaction_type == row['interaction_type']:
                edges.append([user_idx, content_idx])
        
        # Convert to PyTorch tensor and transpose to get (2, num_edges) shape
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def build_graph(self, interactions_df, user_features_df, content_features_df):
        """
        Build a heterogeneous graph from user-content interactions.
        
        Args:
            interactions_df: DataFrame with columns [user_id, content_id, interaction_type, timestamp]
            user_features_df: DataFrame with user features
            content_features_df: DataFrame with content features
            
        Returns:
            PyTorch Geometric Data object
        """
        # Create ID mappings
        self._create_mappings(interactions_df)
        
        # Create edge indices for different interaction types
        view_edge_index = self._create_edge_index(interactions_df, 'view')
        click_edge_index = self._create_edge_index(interactions_df, 'click')
        
        # Process features
        x_user = self._process_user_features(user_features_df)
        x_content = self._process_content_features(content_features_df)
        
        # Create PyTorch Geometric Data object
        data = Data(
            x_user=x_user,
            x_content=x_content,
            edge_index_view=view_edge_index,
            edge_index_click=click_edge_index,
            num_nodes=len(self.user_mapping) + len(self.content_mapping)
        )
        
        return data
    
    def _process_user_features(self, user_features_df):
        """Process user features into tensor format."""
        num_users = len(self.user_mapping)
        feature_dim = self.config.EMBEDDING_DIM
        
        # Initialize feature tensor
        x_user = torch.zeros((num_users, feature_dim))
        
        # Fill in features for each user
        for user_id, features in user_features_df.items():
            if user_id in self.user_mapping:
                idx = self.user_mapping[user_id]
                x_user[idx] = torch.tensor(features[:feature_dim])
                
        return x_user
    
    def _process_content_features(self, content_features_df):
        """Process content features into tensor format."""
        num_content = len(self.content_mapping)
        feature_dim = self.config.EMBEDDING_DIM
        
        # Initialize feature tensor
        x_content = torch.zeros((num_content, feature_dim))
        
        # Fill in features for each content item
        for content_id, features in content_features_df.items():
            if content_id in self.content_mapping:
                idx = self.content_mapping[content_id]
                x_content[idx] = torch.tensor(features[:feature_dim])
                
        return x_content