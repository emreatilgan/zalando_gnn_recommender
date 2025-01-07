import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import MessagePassing

class GNNRecommender(nn.Module):
    def __init__(self, config):
        super(GNNRecommender, self).__init__()
        self.config = config
        
        # Feature aggregators
        self.user_feature_aggregator = nn.LSTM(
            input_size=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_CHANNELS,
            batch_first=True
        )
        
        self.content_feature_aggregator = nn.Linear(
            config.EMBEDDING_DIM,
            config.HIDDEN_CHANNELS
        )
        
        # GNN layers
        self.convs = nn.ModuleList([
            SAGEConv(
                in_channels=(-1, -1) if i == 0 else (config.HIDDEN_CHANNELS, config.HIDDEN_CHANNELS),
                out_channels=config.HIDDEN_CHANNELS
            )
            for i in range(config.NUM_GNN_LAYERS)
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.HIDDEN_CHANNELS * 2, config.HIDDEN_CHANNELS),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_CHANNELS, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_user, x_content, edge_index):
        # Aggregate user features
        user_features, _ = self.user_feature_aggregator(x_user)
        user_features = user_features[:, -1, :]  # Take last LSTM output
        
        # Aggregate content features
        content_features = self.content_feature_aggregator(x_content)
        
        # Combine features
        x = (user_features, content_features)
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        
        # Get embeddings
        user_embedding, content_embedding = x
        
        # Concatenate user and content embeddings
        combined = torch.cat([user_embedding, content_embedding], dim=1)
        
        # Predict probability of click
        return self.classifier(combined)