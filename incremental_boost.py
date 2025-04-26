import numpy as np
import torch
import torch.nn as nn
from utils.proto import Prototype, init_model
from utils.general import *
from utils.prune import model_prune
from utils.get_dataset import *
import argparse
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


def cases_continue(this_y_train,model,num_classes):


    # avoid the case when boosting 
    a = np.unique(this_y_train)
    if len(a)<=1:
        return "Continue"

    # prototype may be zero
    prune_pro_num = 0
    for idx in range(num_classes):
        prune_pro_num += model.class_prototype[idx].shape[1]
        assert model.class_prototype[idx].shape[1] == model.class_transform[idx].shape[1]
    if prune_pro_num==0:
        return "Continue"

    return 0

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


def Prune_model(model, X_train, y_train, feature_min, feature_max, args):
    """Prune the model after training."""
    model.load_parameter(args.save_model + '_model_best.pth')
    model = model_prune(model, len(np.unique(y_train)), X_train, y_train, feature_min, feature_max, args.max_proto_num, args.prune_T)
    model.save_parameter(args.save_model + '_model_prune.pth')
    return model



def Calculate_distance_list(model, this_X_train, this_y_train, feature_min, feature_max, num_classes, feature_num, args):
    '''
    output:
    distance_list: The smaple's distance and feature from each prototype

    Outer list: Represents each class in the model.
    Length: number of classes in the dataset.

    Middle list: Represents each prototype for the current class.
    Length: number of prototypes for the current class.

    Inner list: Represents the distance and feature information for a specific prototype.
    Length: 2

    '''
    distance_list = [[[] for _ in range(model.class_prototype[idx].shape[1])] for idx in range(num_classes)]
            
    idx = 0
    tot = 0
    while idx < this_X_train.shape[0]:
        train_batch = this_X_train[idx:min(idx + args.batch_size, this_X_train.shape[0])]
        train_label = this_y_train[idx:min(idx + args.batch_size, this_y_train.shape[0])]
        
        train_before_norm = train_batch.copy()
        
        train_batch = (train_batch - feature_min) / (feature_max - feature_min)
        
        train_batch = torch.from_numpy(train_batch).float().cuda()
        train_label = torch.from_numpy(train_label).long().cuda()
        
        logits = model(train_batch)
        _, predictions = torch.max(logits, dim=1)
        for dim1 in range(num_classes):
            if model.class_prototype[dim1].shape[1] == 0:
                continue
            
            train_batch = train_batch.view(train_batch.shape[0], 1, feature_num)
            
            # Compute distance via Eq.(1) of Helios
            dist_class = torch.abs(train_batch - model.class_prototype[dim1])
            dist_class = dist_class * torch.abs(model.class_transform[dim1])
            min_dist, _ = torch.max(dist_class, dim=2)
            min_dist, indice = torch.min(min_dist, dim=1)
            
            for dim2 in range(predictions.shape[0]):
                if predictions[dim2] == dim1 and predictions[dim2] == train_label[dim2]:
                    distance_list[dim1][indice[dim2]].append((min_dist[dim2].item(), train_before_norm[dim2]))
                    tot += 1
                    
        idx += args.batch_size

    if tot == 0:
        print("No prototype, stop boosting")
        return "Continue"

    for dim1 in range(num_classes):
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            distance_list[dim1][dim2].sort(reverse=True, key=lambda x: x[0])

    # Distance-based train acc after prune
    # prune_train_acc = tot / this_X_train.shape[0] * 100
    # print('distance-based train acc after prune = %.3f%%' % (prune_train_acc))

    # Distance list of each prototype(sorted and print top 10)
    # for dim1 in range(num_classes):
    #     for dim2 in range(model.class_prototype[dim1].shape[1]):
    #         tmp = [t[0] for t in distance_list[dim1][dim2][:10]]
    #         print(tmp)

    # 
    cost_cnt = 0
    for dim1 in range(num_classes):
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            # x[0] is the distance and x[1] is the feature
            threshold = np.mean(np.array([x[0] for x in distance_list[dim1][dim2]])) * args.threshold_radio

            # Pop the sample above threshold
            while distance_list[dim1][dim2][0][0] > threshold and len(distance_list[dim1][dim2]) > 1:
                distance_list[dim1][dim2].pop(0)
                cost_cnt += 1

    # print('cost due to distance threshold = %d/%d, rate = %.3f%%' % (cost_cnt, tot, cost_cnt / tot * 100))
    return distance_list


def Convert_to_rules(distance_list, model, num_classes, feature_num,):
    '''change prototype to match rules and get final results'''
    temp_class_rules = []
    
    for dim1 in range(num_classes):
        temp_1 = []
        for dim2 in range(model.class_prototype[dim1].shape[1]):
            temp_2 = [[1e10, -1e10] for dim3 in range(feature_num)]
            
            for dim4 in range(len(distance_list[dim1][dim2])):
                sample = distance_list[dim1][dim2][dim4][1]
                for dim5 in range(feature_num):
                    temp_2[dim5][0] = min(temp_2[dim5][0], sample[dim5])
                    temp_2[dim5][1] = max(temp_2[dim5][1], sample[dim5])


            temp_1.append(temp_2)
        temp_class_rules.append(temp_1)

    return temp_class_rules


def Fliter_boosting_residual(class_rules, X_train, y_train, num_classes,):
    cls_fix,_,_ = solve_conflict(X_train, y_train, num_classes, class_rules)
            
    no_match = 0
    tot = 0
    tot_overlap = 0
    overlap_success = 0
    
    fault_list = []

    for idx in range(X_train.shape[0]):
        
        sample = X_train[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    
                    if (dim1 * 10000 + dim2) not in cls_res:
                        cls_res.append(dim1 * 10000 + dim2)
        
        # fix important bug: cls_res[0], not cls_res
        if len(cls_res) == 0:
            no_match += 1
            fault_list.append(idx)
        if len(cls_res) == 1:
            if cls_res[0] // 10000 == y_train[idx]:
                tot += 1
            else:
                fault_list.append(idx)
        elif len(cls_res) >= 2:
            tot_overlap += 1
            cls_res.sort(reverse=False)
            cls_res_hash = '#'
            for p in cls_res:
                cls_res_hash += str(p)
                cls_res_hash += '#'
            if cls_fix[cls_res_hash] == y_train[idx]:
                tot += 1
                overlap_success += 1
            else:
                fault_list.append(idx)
        
    this_X_train = X_train[fault_list]
    this_y_train = y_train[fault_list]
    # print('global rule-based accuracy on train_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_train.shape[0] * 100, tot_overlap, X_train.shape[0], tot_overlap / X_train.shape[0] * 100, overlap_success / X_train.shape[0] * 100, no_match / X_train.shape[0] * 100))
    return this_X_train, this_y_train


def Test_rules(class_rules, unknown_data_list, unknown_label_list, X_test, y_test, num_classes, idx_log_class, attack_classes, log_class):
    '''
    confusion matrix is for the old and new class binary classifications
    acc, f1, recall are indicators for the old category classification
    '''
    #           new old
    # pre_new   tp  fp
    # pre_old   fn  tn

    if len(unknown_data_list) != 0:
        for unknown_class_idx in range(len(unknown_data_list)):
            if unknown_label_list[unknown_class_idx][0]==log_class:
                unknown_data = unknown_data_list[unknown_class_idx]

                # Owing to large differences in the number of categories of data, use micro average
                unknown_label = unknown_label_list[unknown_class_idx]
                fn = 0
                for idx in range(unknown_data.shape[0]):
                    
                    sample = unknown_data[idx]
                    cls_res = []
                    for dim1 in range(num_classes):
                        for dim2 in range(len(class_rules[dim1])):
                            if check_rule(sample, class_rules[dim1][dim2]):
                                
                                if dim1 not in cls_res:
                                    cls_res.append(dim1)
                    if len(cls_res) >= 1:
                        fn += 1

        tp = unknown_data.shape[0] - fn

        # print('unknown class = %d, name = %s, final rule-based detect count = %d/%d, rate = %.3f%%' % (unknown_label[0], attack_classes[unknown_label[0]], unknown_data.shape[0] - fn, unknown_data.shape[0], 100 - fn / unknown_data.shape[0] * 100))

    else:
        tp = 0
        fn = 0
    

    tot = 0
    tot_overlap = 0
    overlap_success = 0
    fp = 0 #预测为新类别，实际为旧类别
    tmp_1 = 0
    y_pred = []
    
    cls_fix,_,_ = solve_conflict(X_test, y_test,num_classes,class_rules)
    for idx in range(X_test.shape[0]):
        
        sample = X_test[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    if (dim1 * 10000 + dim2) not in cls_res:
                        cls_res.append(dim1 * 10000 + dim2)

        # 预测类别只有1个，且旧类别预测正确
        if len(cls_res) == 1:
            y_pred.append(cls_res[0] // 10000)
            if cls_res[0] // 10000 == y_test[idx]:
                tot += 1
            else:
                tmp_1+=1

        # 预测类别有多个，使用矛盾解决后旧类别预测正确
        elif len(cls_res) >= 2:
            tot_overlap += 1
            cls_res.sort(reverse=False)
            cls_res_hash = '#'
            for p in cls_res:
                cls_res_hash += str(p)
                cls_res_hash += '#'
            y_pred.append(cls_fix[cls_res_hash])
            if cls_fix[cls_res_hash] == y_test[idx]:
                tot += 1
                overlap_success += 1
            
        elif len(cls_res) == 0:
            fp += 1
            y_pred.append(num_classes)

    tn = X_test.shape[0] - fp
    # Note that the confusion matrix is for the old and new class binary classifications
    new_cm = [tp,fn,fp,tn]
    final_acc = tot / X_test.shape[0] * 100

    precision = 100* precision_score(y_test, y_pred, average='weighted',zero_division = 0)
    recall = 100* recall_score(y_test, y_pred, average='weighted',zero_division = 0)
    f1 = 100* f1_score(y_test, y_pred, average='weighted',zero_division = 0)


    # print('final rule-based accuracy on test_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%' % (tot / X_test.shape[0] * 100, tot_overlap, X_test.shape[0], tot_overlap / X_test.shape[0] * 100, overlap_success / X_test.shape[0] * 100))
    # print("1_class but wrong rate: ",tmp_1/X_test.shape[0] * 100)
    # print("Confusino Matrix:",new_cm)

    return new_cm, final_acc, precision, recall, f1


def Incremental_rules(class_rules, old_conflict_rules, old_conflict_rules_pri, unknown_data_list, unknown_label_list, X_train, y_train, X_test, y_test, num_classes, idx_log_class, attack_classes, log_class):
    '''Remove rules which contain new sample'''

    old_class_rules = []
    if len(unknown_data_list) != 0:
        for unknown_class_idx in range(len(unknown_data_list)):
            # 只记录log_class
            if unknown_label_list[unknown_class_idx][0]==log_class:

                unknown_data = unknown_data_list[unknown_class_idx]
                unknown_label = unknown_label_list[unknown_class_idx]
                unknown_data, _, unknown_label, _ = train_test_split(unknown_data, unknown_label, test_size=args.test_split_size, random_state=2023)

                this_X_train = unknown_data.copy()
                this_y_train = np.ones_like(unknown_label) * num_classes

                for dim1 in range(num_classes):
                    tmp_old_class_rules = []
                    for dim2 in range(len(class_rules[dim1])):
                        flag = 0
                        for sample in unknown_data:
                            if check_rule(sample, class_rules[dim1][dim2]):
                                flag += 1
                            
                        if flag==0:

                            tmp_old_class_rules.append(class_rules[dim1][dim2])

                    old_class_rules.append(tmp_old_class_rules)


        

        # data plane rule remain for this incremental class
        cls_fix,_,_ = solve_conflict(X_train, y_train,num_classes,class_rules)

        all_conflict_id = [] 
        for key, value in cls_fix.items():
            ids = key.strip('#').split('#')
            all_conflict_id.extend(ids)

        remain_pro_rule = []
        remain_num = 0
        for dim1 in range(num_classes):
            remain_pro_rule_tmp = []
            for dim2 in range(len(old_class_rules[dim1])):
                if str(dim1 * 10000 + dim2) not in all_conflict_id:
                    remain_pro_rule_tmp.append(old_class_rules[dim1][dim2])
                    remain_num += 1
            remain_pro_rule.append(remain_pro_rule_tmp)
                    
        cls_fix,_,_ = solve_conflict(X_train, y_train,num_classes,old_class_rules)
        # reload X
        no_match = 0
        tot = 0
        tot_overlap = 0
        overlap_success = 0
        
        fault_list = []

        for idx in range(X_train.shape[0]):
            
            sample = X_train[idx]
            cls_res = []
            
            for dim1 in range(num_classes):
                for dim2 in range(len(old_class_rules[dim1])):
                    if check_rule(sample, old_class_rules[dim1][dim2]):
                        
                        if (dim1 * 10000 + dim2) not in cls_res:
                            cls_res.append(dim1 * 10000 + dim2)
            
            # fix important bug: cls_res[0], not cls_res
            if len(cls_res) == 0:
                no_match += 1
                fault_list.append(idx)
            if len(cls_res) == 1:
                if cls_res[0] // 10000 == y_train[idx]:
                    tot += 1
                else:
                    fault_list.append(idx)
            elif len(cls_res) >= 2:
                tot_overlap += 1
                cls_res.sort(reverse=False)
                cls_res_hash = '#'
                for p in cls_res:
                    cls_res_hash += str(p)
                    cls_res_hash += '#'
                if cls_fix[cls_res_hash] == y_train[idx]:
                    tot += 1
                    overlap_success += 1
                else:
                    fault_list.append(idx)
            
        print('incremental old rule-based accuracy on train_data = %.3f%%, samples have overlap = %d/%d, rate = %.3f%%, overlap success rate = %.3f%%, no match rate = %.3f%%' % (tot / X_train.shape[0] * 100, tot_overlap, X_train.shape[0], tot_overlap / X_train.shape[0] * 100, overlap_success / X_train.shape[0] * 100, no_match / X_train.shape[0] * 100))
        # for next new class
        this_X_train = np.concatenate((this_X_train, X_train[fault_list]), axis=0)
        this_y_train = np.concatenate((this_y_train, y_train[fault_list]), axis=0)

    else:
        this_X_train = None
        this_y_train = None
        remain_num = -1
        remain_pro_rule = [[] for i in range(num_classes)]


    # Get Conflict_rules and their priorities
    cls_fix,reduced_cls_fix,conflict_rule_num = solve_conflict(X_test, y_test,num_classes,class_rules,topo_sort=True)
    conflict_rules,conflict_rules_pri = conflict_2rule(reduced_cls_fix,class_rules,num_classes)

    # 增量保留重叠：保留下来的原型，同时出现上次规则中
    remain_conflict_rules = [[] for i in range(num_classes-1)]
    remain_conflict_rules_pri = [[] for i in range(num_classes-1)]
    remain_conflict_rules_num = 0

    if idx_log_class != 0:
        for dim1 in range(num_classes-1):
            for dim2 in range(len(conflict_rules[dim1])):

                if conflict_rules[dim1][dim2] in old_conflict_rules[dim1] and conflict_rules_pri[dim1][dim2] in old_conflict_rules_pri[dim1]:
                    remain_conflict_rules[dim1].append(conflict_rules[dim1][dim2])
                    remain_conflict_rules_pri.append(conflict_rules_pri[dim1][dim2])
                    remain_conflict_rules_num += 1

        last_conflict_rules = copy.deepcopy(old_conflict_rules)
        last_conflict_rules_pri = copy.deepcopy(old_conflict_rules_pri)
    else:
        last_conflict_rules = []
        last_conflict_rules_pri = []

    old_conflict_rules_pri = copy.deepcopy(conflict_rules_pri)
    old_conflict_rules = copy.deepcopy(conflict_rules)

    
    # print('solve match conflict:')
    # print(cls_fix)
    
    return this_X_train, this_y_train,\
            old_class_rules, remain_pro_rule, conflict_rules, conflict_rules_pri, last_conflict_rules, last_conflict_rules_pri, remain_conflict_rules, remain_conflict_rules_pri,\
            old_conflict_rules, old_conflict_rules_pri


def record_results(new_cm, final_acc, precision,recall,f1, output_csv, attack_classes, log_class):
    choose_FPR = 100*new_cm[2]/(new_cm[2]+new_cm[3])
    # tp,fn,fp,tn对应相加
    detection_acc = 100*(new_cm[0]+new_cm[3])/sum(new_cm)
    if log_class == -1:
        choose_TPR = -1
        log_class_name = "ALL"    
    else:
        
        choose_TPR = 100*new_cm[0]/(new_cm[0]+new_cm[1])
        log_class_name = attack_classes[log_class]

    experi_csv = [args.dataset,log_class_name,args.threshold_radio,args.boost_num,args.temperature,args.prune_T,args.prune_rule,args.dbscan_eps,detection_acc,choose_TPR,choose_FPR,final_acc,\
                    precision,recall,f1]


            
    if os.path.exists(output_csv):
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            # 写入超参数和结果
            writer.writerow(experi_csv)

def Prune_rules(class_rules, X_train, num_classes, args):
    for dim1 in range(num_classes):
        rules_to_delete = []  # 存储需要删除的规则索引
        for dim2 in range(len(class_rules[dim1])):
            # 规则剪枝
            sample_num = 0
            for sample in X_train:
                if check_rule(sample, class_rules[dim1][dim2]):
                    sample_num += 1
                
            if sample_num <= args.prune_rule:
                rules_to_delete.append(dim2)
        # 删除存储的需要删除的规则，注意要反向删除，以防止索引错误
        for rule_index in sorted(rules_to_delete, reverse=True):
            del class_rules[dim1][rule_index]
    return class_rules


def main(args, output_csv):
    # Set the GPU device
    torch.cuda.set_device(args.gpu)
    set_global_seed(args.seed)
    start_time = time.time()


    # Incremental start
    for idx_log_class, log_class in enumerate(args.add_order):
        # Get data 
        X_train, X_test, y_train, y_test, feature_min, feature_max, unknown_data_list, unknown_label_list, num_classes, feature_num, attack_classes = get_data(args)
        print(f'----------add class: {attack_classes[log_class] if log_class != -1 else "ALL"} ----------')

        if idx_log_class == 0:
            # initial for boosting and class incremental
            this_X_train = X_train.copy()
            this_y_train = y_train.copy()
            class_rules = []
            last_remain_pro_rule = []
            last_class_rules = []
            last_conflict_rules = []
            last_conflict_rules_pri = []
            old_conflict_rules = []
            old_conflict_rules_pri = []
        else:
            last_class_rules = copy.deepcopy(class_rules)
            last_remain_pro_rule = copy.deepcopy(remain_pro_rule)
            class_rules = old_class_rules



        # Boosting start
        for boost_iter in range(args.boost_num):
            # Initialize the model
            model = initialize_model(args, num_classes, this_X_train, this_y_train, feature_min, feature_max)

            # Two cases that should skip
            if cases_continue(this_y_train,model,num_classes)=="Continue":
                continue

            # Train the model
            best_train_acc = Train_model(args, model, this_X_train, this_y_train, feature_min, feature_max)

            # Prune the model
            model = Prune_model(model, this_X_train, this_y_train, feature_min, feature_max, args)
            # if boost_iter==0:
            #     prune_acc = test_model_acc(model, X_test, y_test, feature_min, feature_max)
            #     print('prune_T = %d, distance-based test acc after prune: %.3f%%' % (args.prune_T, prune_acc))

            # Calculate thresholds for each prototype
            distance_list = Calculate_distance_list(model, this_X_train, this_y_train, feature_min, feature_max, num_classes, feature_num, args)
            if distance_list == "Continue":
                continue
            
            # Compute boundaries (range rules) via Eq.(6) of Helios
            temp_class_rules = Convert_to_rules(distance_list, model, num_classes, feature_num,)

            # merge old_class_rules generatied last iter
            class_rules = merge_rules(class_rules, temp_class_rules)

            # Filtering the residual samples
            this_X_train, this_y_train  = Fliter_boosting_residual(class_rules, X_train, y_train, num_classes,)
            
        # Boosting end 


        if args.prune_rule != 0:
            class_rules = Prune_rules(class_rules, X_train, num_classes, args)

        # Record_results
        new_cm, final_acc, precision, recall, f1 = Test_rules(class_rules, unknown_data_list, unknown_label_list, X_test, y_test, num_classes, idx_log_class, attack_classes, log_class)
        record_results(new_cm, final_acc, precision, recall, f1, output_csv, attack_classes, log_class)

        # Increment rules: remove rules that include new sample
        this_X_train, this_y_train,\
        old_class_rules, remain_pro_rule,\
        conflict_rules, conflict_rules_pri,\
        last_conflict_rules, last_conflict_rules_pri,\
        remain_conflict_rules, remain_conflict_rules_pri,\
        old_conflict_rules, old_conflict_rules_pri = Incremental_rules(class_rules, old_conflict_rules, old_conflict_rules_pri, unknown_data_list, unknown_label_list, X_train, y_train, X_test, y_test, num_classes, idx_log_class, attack_classes, log_class)


        args.selected_class.append(log_class)

    # Incremental end
        
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
    parser.add_argument('--prune_T', type=int, default=10, help='The number of samples that minimally correspond to each prototype, controls the level of pruning after training.')
    parser.add_argument('--prune_rule', type=int, default=5, help='The number of samples that minimally correspond to each rule, rule pruning threshold, similar to prune_T but different')
    parser.add_argument('--threshold_radio', type=float, default=1.2, help='a radio of a multiplied by boundaries, controls boundaries of prototypes.')
    parser.add_argument('--init_type', type=str, default='DBSCAN', help='Initialization method (e.g., DBSCAN, KMEANS)')
    parser.add_argument('--dbscan_eps', type=float, default=0.1, help='(if --init_type is DBSCAN) Epsilon value for clustering, controls the size of neighborhood for clustering.')
    parser.add_argument('--min_samples', type=int, default=2, help='(if --init_type is DBSCAN) minimum samples for forming a cluster, controls the density of clusters.')
    parser.add_argument('--boost_num', type=int, default=4, help='Number of boosting iterations, controls how many boosting steps will be performed.')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--dataset', type=str, default='cicids-2018', help='Dataset name, defining in get_data.py')
    parser.add_argument('--attack_max_samples', type=int, default=10000, help='Maximum number of samples per attack class in the dataset.')
    parser.add_argument('--selected_class', type=list, default=[0, 1], help='List of known attack classes for training, defines the subset of attack classes to be used.')
    parser.add_argument('--add_order', type=list, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, -1], help='Order in which classes are added for incremental learning(Unknown class), -1 indicates no new class.')
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
