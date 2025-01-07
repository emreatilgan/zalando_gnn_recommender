import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from typing import Optional, Tuple

class UserFeatureAggregator(nn.Module):
    """
    Aggregates user features using LSTM layer as shown in the architecture diagram.
    Processes sequential data of user's recent orders.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super(UserFeatureAggregator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer for processing sequential order history
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Additional processing layers
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: User features tensor of shape (batch_size, seq_len, input_dim)
            lengths: Optional tensor of sequence lengths for packed sequence
            
        Returns:
            Aggregated user features of shape (batch_size, hidden_dim)
        """
        if lengths is not None:
            # Pack the sequence for variable length processing
            packed_input = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_output, (hidden, _) = self.lstm(packed_input)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, _) = self.lstm(x)
        
        # Get the last output for each sequence
        last_output = output[:, -1, :]
        
        # Apply normalization and dropout
        last_output = self.batch_norm(last_output)
        last_output = self.dropout(last_output)
        
        return last_output

class ContentFeatureAggregator(nn.Module):
    """
    Aggregates content features using mean aggregation as shown in the architecture diagram.
    Processes article representations associated with each content piece.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super(ContentFeatureAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Layers for processing aggregated features
        self.transform = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Content features tensor of shape (batch_size, num_articles, input_dim)
            mask: Optional boolean mask for valid articles
            
        Returns:
            Aggregated content features of shape (batch_size, output_dim)
        """
        if mask is not None:
            # Apply mask and compute mean over valid articles
            masked_x = x * mask.unsqueeze(-1)
            sum_x = masked_x.sum(dim=1)
            count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_x = sum_x / count
        else:
            # Simple mean over all articles
            mean_x = x.mean(dim=1)
        
        # Transform aggregated features
        output = self.transform(mean_x)
        return output

class FeatureAggregationLayer(MessagePassing):
    """
    Custom message passing layer that combines both user and content feature aggregation.
    Implements the neighborhood feature aggregation shown in Image 3.
    """
    def __init__(self, user_dim: int, content_dim: int, output_dim: int):
        super(FeatureAggregationLayer, self).__init__(aggr='mean')
        self.user_dim = user_dim
        self.content_dim = content_dim
        self.output_dim = output_dim
        
        # Feature aggregators
        self.user_aggregator = UserFeatureAggregator(user_dim, output_dim)
        self.content_aggregator = ContentFeatureAggregator(content_dim, output_dim)
        
        # Final transformation for combined features
        self.combine = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, 
                x_user: torch.Tensor,
                x_content: torch.Tensor,
                edge_index: torch.Tensor,
                user_lengths: Optional[torch.Tensor] = None,
                content_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_user: User features
            x_content: Content features
            edge_index: Graph connectivity
            user_lengths: Optional lengths of user sequences
            content_mask: Optional mask for valid articles
            
        Returns:
            Tuple of aggregated user and content features
        """
        # Aggregate user and content features separately
        user_features = self.user_aggregator(x_user, user_lengths)
        content_features = self.content_aggregator(x_content, content_mask)
        
        # Propagate features through the graph
        out_user = self.propagate(edge_index, x=(user_features, content_features))
        out_content = self.propagate(edge_index.flip([0]), x=(content_features, user_features))
        
        # Combine propagated features
        user_combined = self.combine(torch.cat([user_features, out_user], dim=1))
        content_combined = self.combine(torch.cat([content_features, out_content], dim=1))
        
        return user_combined, content_combined
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Constructs messages to be aggregated."""
        return x_j