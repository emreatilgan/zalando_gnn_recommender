from sklearn.metrics import roc_auc_score, average_precision_score

def calculate_metrics(predictions, labels):
    """Calculate evaluation metrics"""
    return {
        'roc_auc': roc_auc_score(labels, predictions),
        'average_precision': average_precision_score(labels, predictions)
    }