class Config:
    # Graph parameters
    NUM_GNN_LAYERS = 2
    HIDDEN_CHANNELS = 64
    EMBEDDING_DIM = 32
    
    # Training parameters
    BATCH_SIZE = 512
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # Data parameters
    NUM_NEIGHBORS = [10, 10]  # Number of neighbors to sample for each GNN layer
    
    # Model parameters
    DROPOUT = 0.2
    
    # Paths
    DATA_PATH = "data/"
    CHECKPOINTS_PATH = "checkpoints/"