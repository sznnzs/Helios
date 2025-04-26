import numpy as np
import torch
import torch.nn as nn

# Global configuration
BATCH_SIZE = 512  # Processing batch size for memory efficiency

def model_prune(model, num_classes, X_train, y_train, feature_min, feature_max, max_proto_num, prune_T):
    """
    Prune prototypes based on their support counts from training data.
    
    Args:
        model: Prototype model to prune
        num_classes: Number of target classes
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        feature_min: Minimum feature values for normalization
        feature_max: Maximum feature values for normalization
        max_proto_num: Maximum allowed prototypes per class
        prune_T: Threshold for prototype pruning (prototypes with support <= T will be removed)
        
    Returns:
        Pruned model
    """
    
    # Initialize support counter matrix: [class_id][prototype_id] -> support_count
    support_num = np.zeros((num_classes, max_proto_num), dtype=np.int64)
    feature_dim = len(feature_min)

    # Batch processing of training data
    idx = 0
    while idx < X_train.shape[0]:
        # Get current batch
        batch_end = min(idx + BATCH_SIZE, X_train.shape[0])
        train_batch = X_train[idx:batch_end]
        train_label = y_train[idx:batch_end]
        
        # Normalize features to [0, 1]
        train_batch = (train_batch - feature_min) / (feature_max - feature_min)
        
        # Convert to tensors and move to GPU
        train_batch = torch.from_numpy(train_batch).float().cuda()
        train_label = torch.from_numpy(train_label).long().cuda()
        
        # Get model predictions
        with torch.no_grad():
            logits = model(train_batch)
        _, predictions = torch.max(logits, dim=1)

        # Calculate prototype supports for each class
        for class_id in range(num_classes):
            # Skip classes without prototypes
            if model.class_prototype[class_id].shape[1] == 0:
                continue
            
            # Reshape for broadcasting
            batch_view = train_batch.view(train_batch.shape[0], 1, feature_dim)
            
            if model.cal_dis == 'l_n':
                # L-infinity distance calculation
                if model.cal_mode == 'trans_abs':
                    transformed_batch = batch_view * torch.abs(model.class_transform[class_id])
                    dist_matrix = torch.abs(transformed_batch - model.class_prototype[class_id])
                elif model.cal_mode == 'abs_trans':
                    dist_matrix = torch.abs(batch_view - model.class_prototype[class_id])
                    dist_matrix *= torch.abs(model.class_transform[class_id])
                
                # Get min-max distances
                max_dist, _ = torch.max(dist_matrix, dim=2)
                min_dist, indices = torch.min(max_dist, dim=1)
                
            else:  # L2 distance calculation
                dist_matrix = (batch_view - model.class_prototype[class_id]) ** 2
                dist_matrix = torch.sum(dist_matrix, dim=2)
                min_dist, indices = torch.min(dist_matrix, dim=1)
            
            # Update support counts for correctly classified samples
            for sample_idx in range(predictions.shape[0]):
                if predictions[sample_idx] == class_id and predictions[sample_idx] == train_label[sample_idx]:
                    prototype_idx = indices[sample_idx].item()
                    support_num[class_id][prototype_idx] += 1

        idx += BATCH_SIZE

    # Print pruning statistics
    total_support = np.sum(support_num)
    print(f'Train samples: {X_train.shape[0]}, Total supports: {total_support}, '
          f'Pre-prune accuracy: {total_support/X_train.shape[0]*100:.3f}%')

    # Perform prototype pruning
    for class_id in range(num_classes):
        # Sort prototypes by support count
        prototype_supports = {pid: support_num[class_id][pid] for pid in range(support_num.shape[1])}
        sorted_supports = sorted(prototype_supports.items(), key=lambda x: x[1])
        
        # Identify prototypes to remove
        remove_ids = []
        valid_supports = []
        
        for pid, support in sorted_supports:
            # Skip invalid prototype indices
            if pid >= model.class_prototype[class_id].shape[1]:
                continue
            
            if support <= prune_T:
                remove_ids.append(pid)
            else:
                valid_supports.append(support)
        
        # Remove low-support prototypes
        if remove_ids:
            # Update prototype parameters
            new_prototypes = np.delete(
                model.class_prototype[class_id].detach().cpu().numpy(),
                remove_ids,
                axis=1
            )
            model.class_prototype[class_id] = nn.Parameter(torch.from_numpy(new_prototypes).cuda())
            
            # Update transformation parameters
            new_transforms = np.delete(
                model.class_transform[class_id].detach().cpu().numpy(),
                remove_ids,
                axis=1
            )
            model.class_transform[class_id] = nn.Parameter(torch.from_numpy(new_transforms).cuda())
            
            # Verify parameter consistency
            assert model.class_prototype[class_id].shape == model.class_transform[class_id].shape

    return model