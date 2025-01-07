import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
import os
import logging
from datetime import datetime

from models.gnn import GNNRecommender
from data.dataset import ZalandoDataset, create_dataloader
from utils.metrics import calculate_metrics
from config import Config

def setup_logging(config):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(config.CHECKPOINTS_PATH, f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return log_dir

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    labels = []
    
    progress_bar = tqdm(loader, desc='Training')
    for batch in progress_bar:
        # Move batch to device
        batch = batch.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(
            batch.x_user,
            batch.x_content,
            batch.edge_index_view
        )
        
        # Calculate loss
        labels_batch = batch.edge_index_click is not None
        loss = criterion(pred, labels_batch.float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Store predictions and labels for metrics
        total_loss += loss.item() * batch.num_graphs
        predictions.extend(pred.detach().cpu().numpy())
        labels.extend(labels_batch.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(predictions),
        np.array(labels)
    )
    metrics['loss'] = total_loss / len(loader.dataset)
    
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    labels = []
    
    for batch in tqdm(loader, desc='Validation'):
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass
        pred = model(
            batch.x_user,
            batch.x_content,
            batch.edge_index_view
        )
        
        # Calculate loss
        labels_batch = batch.edge_index_click is not None
        loss = criterion(pred, labels_batch.float())
        
        # Store predictions and labels for metrics
        total_loss += loss.item() * batch.num_graphs
        predictions.extend(pred.cpu().numpy())
        labels.extend(labels_batch.cpu().numpy())
    
    # Calculate metrics
    metrics = calculate_metrics(
        np.array(predictions),
        np.array(labels)
    )
    metrics['loss'] = total_loss / len(loader.dataset)
    
    return metrics

def train(config):
    """Main training function"""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_dir = setup_logging(config)
    writer = SummaryWriter(log_dir)
    
    # Create dataset and dataloaders
    dataset = ZalandoDataset(root=config.DATA_PATH, config=config)
    train_loader = create_dataloader(dataset, config, mode='train')
    val_loader = create_dataloader(dataset, config, mode='val')
    
    # Initialize model
    model = GNNRecommender(config).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_auc = 0
    for epoch in range(config.NUM_EPOCHS):
        logging.info(f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Log metrics
        for name, value in train_metrics.items():
            writer.add_scalar(f'train/{name}', value, epoch)
        for name, value in val_metrics.items():
            writer.add_scalar(f'val/{name}', value, epoch)
        
        # Update learning rate
        scheduler.step(val_metrics['roc_auc'])
        
        # Save best model
        if val_metrics['roc_auc'] > best_val_auc:
            best_val_auc = val_metrics['roc_auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': best_val_auc,
                'config': config,
            }, os.path.join(log_dir, 'best_model.pt'))
        
        # Log epoch summary
        logging.info(
            f'Train Loss: {train_metrics["loss"]:.4f} '
            f'Train AUC: {train_metrics["roc_auc"]:.4f} '
            f'Val Loss: {val_metrics["loss"]:.4f} '
            f'Val AUC: {val_metrics["roc_auc"]:.4f}'
        )
    
    writer.close()
    return model, best_val_auc

if __name__ == '__main__':
    config = Config()
    model, best_val_auc = train(config)
    logging.info(f'Training completed. Best validation AUC: {best_val_auc:.4f}')