import numpy as np
from sklearn.model_selection import train_test_split

def get_data(args):
    """
    This function loads the dataset based on the user's choice and returns the training and testing data.
    It supports two datasets: 'cicids-2018' and 'unsw-nb15'.
    """
    
    if args.dataset == 'cicids-2018':
        # List of attack classes in CICIDS-2018 dataset, including 'BENIGN' as the normal class
        attack_classes = ['BENIGN', 'DDoS_LOIC_HTTP', 'DDoS_HOIC', 'DDoS_LOIC_UDP', 
                          'DoS_GoldenEye', 'DoS_Hulk', 'DoS_Slowloris', 
                          'SSH_BruteForce', 'Web_Attack_XSS', 'Web_Attack_SQL', 'Web_Attack_Brute_Force']
        
        # The number of classes selected for training (based on user's selection)
        num_classes = len(args.selected_class)

        # Load the CICIDS-2018 data, splitting into training and testing sets
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list = \
            get_data_CICIDS_2018(args.attack_max_samples, args.selected_class, args.test_split_size)

    # Get the number of features from the training data (columns)
    feature_num = X_train.shape[1]

    # Ensure that the feature_min and feature_max have the same number of dimensions as the feature data
    assert feature_num == len(feature_min)
    assert feature_num == len(feature_max)

    # Return the processed data and metadata
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list, num_classes, feature_num, attack_classes


def get_data_CICIDS_2018(iscx_attack_max_samples, iscx_selected_class, test_split_size):
    """
    This function loads and processes the CICIDS-2018 dataset. It selects data based on the user's class selection
    and splits it into training and testing sets.
    """
    
    # Load the features and labels for the CICIDS-2018 dataset
    data = np.load("./dataset/CICIDS_2018_X.npy").astype(np.int64)
    label = np.load("./dataset/CICIDS_2018_y.npy").astype(np.int64)

    selected_data = []
    selected_label = []

    # Loop over the selected attack classes and gather the corresponding data
    for idx, attack_class in enumerate(iscx_selected_class):
        # Select up to 'iscx_attack_max_samples' samples from each class
        selected_data.append(data[label == attack_class][:iscx_attack_max_samples])
        
        # Create labels corresponding to the class
        length = data[label == attack_class][:iscx_attack_max_samples].shape[0]
        selected_label.append(np.ones(length) * idx)
    
    # Stack the selected data and labels into one dataset
    selected_data = np.vstack(selected_data)
    selected_label = np.hstack(selected_label)

    # Calculate the minimum and maximum values for feature scaling
    feature_max = np.max(selected_data, axis=0)
    feature_min = np.min(selected_data, axis=0)

    # If any feature has the same min and max value (i.e., constant), fix the values to avoid division by zero
    if 0 in (feature_max - feature_min):
        idx0 = (feature_max - feature_min) == 0
        feature_min[idx0] = 0
        feature_max[idx0] = 1

    # Split the selected data into training and testing sets using the specified test split size
    X_train, X_test, y_train, y_test = train_test_split(selected_data, selected_label, test_size=test_split_size, random_state=2024)

    # Prepare lists to store data from unknown attack classes (those not selected for training)
    unknown_data_list = []
    unknown_label_list = []

    # Loop through all attack classes and gather data for unknown classes
    for dim1 in range(len(np.unique(label))):
        if dim1 in iscx_selected_class:
            continue
        
        # Collect data for the unknown class
        new_class_index = []
        cnt = 0
        for dim2 in range(label.shape[0]):
            if label[dim2] == dim1:
                new_class_index.append(dim2)
                cnt += 1
                if cnt == iscx_attack_max_samples:
                    break
        unknown_data_list.append(data[new_class_index])
        unknown_label_list.append(label[new_class_index])

    # Return the processed data and metadata
    return X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list

