import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
import torch
import torch.nn.functional as F

def compute_metrics(eval_pred):
    """
    Custom metrics computation function that calculates log loss (cross-entropy) as primary metric,
    along with accuracy, precision, recall, and F1 score.
    
    Args:
        eval_pred (tuple): Tuple containing predictions (logits) and labels from the model
        
    Returns:
        dict: Dictionary containing computed metrics with log_loss as the primary metric
    """
    logits, labels = eval_pred
    
    # Convert logits to probabilities using softmax
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
    
    # Calculate log loss (cross-entropy)
    try:
        loss = log_loss(labels, probs, labels=np.arange(probs.shape[1]))
    except ValueError:
        # Handle edge cases where some classes might not be present
        loss = float('inf')
    
    # Get predicted classes for other metrics
    predictions = np.argmax(logits, axis=-1)
    
    # Calculate additional metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'log_loss': loss,  # Primary metric for optimization
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    } 