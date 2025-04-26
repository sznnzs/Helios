import torch.nn as nn
import torch
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

# Available distance calculation methods: ['l_2', 'l_n']
# Available calculation modes: ['trans_abs', 'abs_trans']

class Prototype(nn.Module):
    
    def __init__(self, num_classes, feature_num, prototype_num_classes, temperature, cal_dis, cal_mode):
        super(Prototype, self).__init__()
        
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.temperature = temperature
        self.cal_dis = cal_dis  # Distance calculation method
        self.cal_mode = cal_mode  # Transformation mode
        
        # Initialize prototype and transformation parameters for each class
        self.class_prototype = []
        self.class_transform = []
        for i in range(self.num_classes):
            self.class_prototype.append(nn.Parameter(torch.randn(1, prototype_num_classes[i], self.feature_num).cuda()))
            self.class_transform.append(nn.Parameter(torch.randn(1, prototype_num_classes[i], self.feature_num).cuda()))
        
        # Uniformly initialize prototypes between 0 and 1
        for i in range(self.num_classes):
            torch.nn.init.uniform_(self.class_prototype[i], a=0, b=1)
        
    def forward(self, train_batch):
        # Reshape input tensor to [batch_size, 1, feature_dim]
        train_batch = train_batch.view(train_batch.shape[0], 1, self.feature_num)
        
        min_dist_class = []
        
        if self.cal_dis == 'l_2':
            # Calculate L2 distance between samples and class prototypes
            for idx in range(self.num_classes):
                if self.class_prototype[idx].shape[1] == 0:  # Handle empty prototypes
                    inf_dist = torch.zeros(train_batch.shape[0]).cuda()
                    min_dist_class.append(inf_dist + 10000)
                    continue
                
                dist_class = (train_batch - self.class_prototype[idx]) ** 2
                dist_class = torch.sum(dist_class, dim=2)
                min_dist, _ = torch.min(dist_class, dim=1)
                min_dist_class.append(min_dist)
            
        elif self.cal_dis == 'l_n':
            # Calculate L-infinity distance with transformation
            for idx in range(self.num_classes):
                if self.class_prototype[idx].shape[1] == 0:  # Handle empty prototypes
                    inf_dist = torch.zeros(train_batch.shape[0]).cuda()
                    min_dist_class.append(inf_dist + 10000)
                    continue
                
                if self.cal_mode == 'trans_abs':
                    # Apply transformation before absolute value
                    train_batch = train_batch * torch.abs(self.class_transform[idx])
                    dist_class = torch.abs(train_batch - self.class_prototype[idx])
                
                elif self.cal_mode == 'abs_trans':
                    # Apply absolute value before transformation
                    dist_class = torch.abs(train_batch - self.class_prototype[idx])
                    dist_class = dist_class * torch.abs(self.class_transform[idx])

                # Get max distance across features then min across prototypes
                min_dist, _ = torch.max(dist_class, dim=2)
                min_dist, _ = torch.min(min_dist, dim=1)
                min_dist_class.append(min_dist)
        
        # Calculate similarity scores
        class_similar_score = []
        for idx in range(self.num_classes):
            score = 1 / (min_dist_class[idx] + 1e-6) * self.temperature 
            class_similar_score.append(score)
        
        # Concatenate scores and apply softmax
        for idx in range(self.num_classes):
            class_similar_score[idx] = class_similar_score[idx].view(-1, 1)
        
        min_dist = torch.cat(class_similar_score, dim=1)
        logit = nn.functional.softmax(min_dist, dim=1)
        
        return logit
    
    def save_parameter(self, dir):
        """Save model parameters to file"""
        torch.save(self.class_prototype + self.class_transform, dir)
    
    def load_parameter(self, dir):
        """Load parameters from file, handling possible class increments"""
        state = torch.load(dir, map_location='cpu')
        
        # Load prototypes
        for i in range(min(self.num_classes, len(state) // 2)):
            if self.class_prototype[i].shape[1] != state[i].shape[1]:
                self.class_prototype[i] = nn.Parameter(torch.FloatTensor(1, state[i].shape[1], self.feature_num).cuda())
            self.class_prototype[i].data = state[i].cuda().data
        
        # Load transformations
        for i in range(min(self.num_classes, len(state) // 2)):
            if self.class_transform[i].shape[1] != state[i + len(state) // 2].shape[1]:
                self.class_transform[i] = nn.Parameter(torch.FloatTensor(1, state[i + len(state) // 2].shape[1], self.feature_num).cuda())
            self.class_transform[i].data = state[i + len(state) // 2].cuda().data


def init_model(model, args, init_type, dbscan_eps, min_samples, X_train, y_train, feature_min, feature_max):
    """Initialize model prototypes using clustering algorithms"""
    
    if init_type == 'NONE':
        return model
    
    elif init_type == 'DBSCAN':
        results = {}
        centers = {}
        noise_counts = {}

        # Process each class separately
        unique_labels = np.unique(y_train)
        
        for label in unique_labels:
            # Get class samples and normalize
            class_mask = (y_train == label)
            X_class = X_train[class_mask]
            X_class_scaled = (X_class - feature_min) / (feature_max - feature_min)

            # Apply DBSCAN clustering
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(X_class_scaled)

            # Store clustering results
            results[label] = cluster_labels
            centers[label] = {}
            
            # Calculate cluster centers
            unique_clusters = set(cluster_labels)
            for cluster in unique_clusters:
                if cluster != -1:  # Ignore noise points
                    cluster_mask = (cluster_labels == cluster)
                    cluster_points = X_class_scaled[cluster_mask]
                    cluster_center = np.mean(cluster_points, axis=0)
                    centers[label][cluster] = {
                        'center': cluster_center,
                        'count': cluster_points.shape[0]
                    }

            # Count noise points
            noise_counts[label] = np.sum(cluster_labels == -1)
        
        # Initialize model parameters based on clustering results
        model.class_prototype = []
        model.class_transform = []

        for i in range(model.num_classes):
            if i not in centers:
                # Initialize empty prototypes for missing classes
                model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, 0, model.feature_num).cuda()))
                model.class_transform.append(nn.Parameter(torch.zeros(1, 0, model.feature_num).cuda() + 1))
                continue
        
            # Initialize prototypes and transformations
            num_clusters = len(centers[i])
            model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, num_clusters, model.feature_num).cuda()))
            model.class_transform.append(nn.Parameter(torch.zeros(1, num_clusters, model.feature_num).cuda() + 1))
        
        # Copy cluster centers to prototype parameters
        for i in range(model.num_classes):
            if i not in centers:
                continue
            for idx, cluster in enumerate(centers[i]):
                with torch.no_grad():
                    model.class_prototype[i][0][idx].copy_(torch.tensor(centers[i][cluster]['center']))
        
        return model

    elif init_type == 'KMEANS':
        results = {}
        centers = {}
        noise_counts = {}

        # Process each class separately
        unique_labels = np.unique(y_train)

        for label in unique_labels:
            # Get class samples and normalize
            class_mask = (y_train == label)
            X_class = X_train[class_mask]
            X_class_scaled = (X_class - feature_min) / (feature_max - feature_min)
            
            # Apply KMeans clustering
            n_clusters = min(100, X_class_scaled.shape[0])
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            cluster_labels = kmeans.fit_predict(X_class_scaled)

            # Store clustering results
            results[label] = cluster_labels
            centers[label] = {}
            
            # Calculate cluster centers
            unique_clusters = set(cluster_labels)
            for cluster in unique_clusters:
                cluster_mask = (cluster_labels == cluster)
                cluster_points = X_class_scaled[cluster_mask]
                cluster_center = np.mean(cluster_points, axis=0)
                centers[label][cluster] = {
                    'center': cluster_center,
                    'count': cluster_points.shape[0]
                }

            # KMeans doesn't produce noise points
            noise_counts[label] = 0

        # Initialize model parameters based on clustering results
        model.class_prototype = []
        model.class_transform = []

        for i in range(model.num_classes):
            if i not in centers:
                # Initialize empty prototypes for missing classes
                model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, 0, model.feature_num).cuda()))
                model.class_transform.append(nn.Parameter(torch.zeros(1, 0, model.feature_num).cuda() + 1))
                continue
        
            # Initialize prototypes and transformations
            num_clusters = len(centers[i])
            model.class_prototype.append(nn.Parameter(torch.FloatTensor(1, num_clusters, model.feature_num).cuda()))
            model.class_transform.append(nn.Parameter(torch.zeros(1, num_clusters, model.feature_num).cuda() + 1))

        # Copy cluster centers to prototype parameters
        for i in range(model.num_classes):
            if i not in centers:
                continue
            for idx, cluster in enumerate(centers[i]):
                with torch.no_grad():
                    model.class_prototype[i][0][idx].copy_(torch.tensor(centers[i][cluster]['center']))

        return model