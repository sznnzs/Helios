import numpy as np
import torch
import torch.nn as nn
from utils.proto import Prototype, init_model
from utils.general import *
from utils.prune import model_prune
from utils.get_dataset import *
import argparse
import wandb
import os
import csv
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time


def initialize_model(args, num_classes, this_X_train, this_y_train, feature_min, feature_max):
    feature_num = this_X_train.shape[1]

    model = Prototype(num_classes, feature_num, [args.max_proto_num for _ in range(num_classes)], args.temperature, "l_n", "abs_trans").cuda()

    model = init_model(model, args, args.init_type, args.dbscan_eps, args.min_samples, this_X_train, this_y_train, feature_min, feature_max)

    return model


def Train_model(args, model, X_train, y_train, feature_min, feature_max):
    """Train the model on the training data."""
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.class_prototype + model.class_transform, lr=args.learning_rate)
    # train_acc = test_model_acc(model, this_X_train, this_y_train, feature_min, feature_max)
    # print('before training, train acc = ', train_acc)
    best_train_acc = 0
    for epoch in range(args.epochs):
        train_acc_num = 0
        idx = 0

        while idx < X_train.shape[0]:
            train_batch = X_train[idx: min(idx + args.batch_size, X_train.shape[0])]
            train_label = y_train[idx: min(idx + args.batch_size, y_train.shape[0])]
            train_batch = (train_batch - feature_min) / (feature_max - feature_min)
            train_batch = torch.from_numpy(train_batch).float().cuda()
            train_label = torch.from_numpy(train_label).long().cuda()

            optimizer.zero_grad()
            logits = model(train_batch)
            _, predictions = torch.max(logits, dim=1)
            loss = loss_func(logits, train_label)
            loss.backward()
            optimizer.step()

            train_acc_num += torch.sum(predictions == train_label).item()
            idx += args.batch_size

        train_acc = train_acc_num / X_train.shape[0] * 100
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            model.save_parameter(args.save_model + '_model_best.pth')

    return best_train_acc


def main(args, output_csv):
    # Set the GPU device
    torch.cuda.set_device(args.gpu)
    set_global_seed(args.seed)
    start_time = time.time()

    
    # Get data 
    X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list, num_classes, feature_num, attack_classes = get_data(args)

    # Initialize the model
    model = initialize_model(args, num_classes, X_train, y_train, feature_min, feature_max)

    # Train the model
    best_train_acc = Train_model(args, model, X_train, y_train, feature_min, feature_max)
    
    test_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
    print("ACC on test data = %.3f%%" % (test_acc))


    end_time = time.time()
    print(f'Training time: {end_time - start_time}s')



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Helios")

    # Add all arguments for the experiment
    parser.add_argument('--gpu', type=int, default=0, help='Single GPU device index.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training, defines how many samples per batch during training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training, defines how many times the model will see the entire dataset.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer, controls the size of the steps the model takes during optimization.')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for controlling the softmax function, affecting the scaling of logits during training.')
    parser.add_argument('--max_proto_num', type=int, default=2000, help=' Maximum number of prototypes per class, used in the prototype network.')
    parser.add_argument('--init_type', type=str, default='DBSCAN', help='Initialization method (e.g., DBSCAN, KMEANS)')
    parser.add_argument('--dbscan_eps', type=float, default=0.1, help='(if --init_type is DBSCAN) Epsilon value for clustering, controls the size of neighborhood for clustering.')
    parser.add_argument('--min_samples', type=int, default=2, help='(if --init_type is DBSCAN) minimum samples for forming a cluster, controls the density of clusters.')
    parser.add_argument('--boost_num', type=int, default=4, help='Number of boosting iterations, controls how many boosting steps will be performed.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--dataset', type=str, default='cicids-2018', help='Dataset name, defining in get_data.py')
    parser.add_argument('--attack_max_samples', type=int, default=10000, help='Maximum number of samples per attack class in the dataset.')
    parser.add_argument('--selected_class', type=list, default=[0, 1], help='List of known attack classes for training, defines the subset of attack classes to be used.')
    parser.add_argument('--test_split_size', type=float, default=0.8, help='Split size for the test dataset, used in few-shot learning settings, when few-shot setting, should large.')
    parser.add_argument('--save_model', type=str, default='./checkpoints/cicids', help='Path to save the trained model.')

    args = parser.parse_args()
    output_csv = "results.csv"
    record_list_name = ['dataset','add_order','choose_threshold','boost_num','temperature','prune_T','prune_rule','dbscan_eps',\
                    'detection_acc','choose_TPR','choose_FPR','final_acc','precision','recall','f1']
    # Check if the CSV file exists, if not create and write the table header
    if not os.path.exists(output_csv):
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(record_list_name)

    main(args, output_csv)
