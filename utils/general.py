import torch
import torch.nn as nn
import numpy as np
import random
import statistics
from collections import Counter
import copy

batch_size = 512

def merge_rules(c, d):
    """Merge two rule lists ensuring there are no duplicates, with rules from the second list having higher priority."""
    a = copy.deepcopy(c)
    b = copy.deepcopy(d)
    result = []

    seen_rules = set()

    # First, merge rules from `b` (second list) while ensuring no duplicates
    for rules_b in b:
        merged_result_b = []
        for rule in rules_b:
            rule_tuple = tuple(map(tuple, rule))  # Convert rule to tuple for easy comparison
            if rule_tuple not in seen_rules:
                seen_rules.add(rule_tuple)
                merged_result_b.append(rule)
        result.append(merged_result_b)

    # Now, merge rules from `a`, avoiding duplicates that already exist in `b`
    for i, rules_a in enumerate(a):
        if i < len(result):  # If result already contains merged rules, just add more
            for rule in rules_a:
                rule_tuple = tuple(map(tuple, rule))
                if rule_tuple not in seen_rules:  # Only add if it's not a duplicate
                    result[i].append(rule)
        else:
            merged_result_a = []
            for rule in rules_a:
                rule_tuple = tuple(map(tuple, rule))
                if rule_tuple not in seen_rules:
                    seen_rules.add(rule_tuple)
                    merged_result_a.append(rule)
            result.append(merged_result_a)

    # Ensure the output has the same number of categories as the input `a`
    while len(result) < len(a):
        result.append([])

    return result


def remove_rules(class_rules, temp_class_rules):
    """Remove rules that are present in `temp_class_rules` from `class_rules`."""
    class_rules_new = copy.deepcopy(class_rules)
    temp_class_rules_new = copy.deepcopy(temp_class_rules)
    new_class_rules = []
    
    # Iterate over all categories
    for class_idx in range(len(class_rules_new)):
        class_rule = class_rules_new[class_idx]
        temp_rule = temp_class_rules_new[class_idx]
        
        updated_rules = []
        
        # Remove rules in `class_rule` that also exist in `temp_rule`
        for rule in class_rule:
            if rule not in temp_rule:
                updated_rules.append(rule)
        
        # Append the updated rules to the new class rules
        new_class_rules.append(updated_rules)
    
    return new_class_rules

def is_contained(delete_class_rules, delete_conflict_rules):
    """Check if any rule in `delete_conflict_rules` is contained within `delete_class_rules`."""
    def is_rule_contained(rule, candidate):
        if len(rule) != len(candidate):
            return False
        for r, c in zip(rule, candidate):
            if not (r[0] >= c[0] and r[1] <= c[1]):  # Check if the rule is contained
                return False
        return True

    # Iterate through each category
    for class_idx in range(len(delete_conflict_rules)):
        conflict_rules = delete_conflict_rules[class_idx]
        class_rules = delete_class_rules[class_idx]
        
        for c_rule in conflict_rules:
            # Check if the conflict rule is contained in any of the class rules
            contained = any(is_rule_contained(c_rule, d_rule) for d_rule in class_rules)
            if not contained:
                return False
    
    return True

def test_model_acc(model, X_test, y_test, feature_min, feature_max):
    """Test model accuracy by evaluating it on the test dataset."""
    test_acc_num = 0
    idx = 0
    while idx < X_test.shape[0]:
        test_batch = X_test[idx:min(idx + batch_size, X_test.shape[0])]
        test_label = y_test[idx:min(idx + batch_size, y_test.shape[0])]
        
        test_batch = (test_batch - feature_min) / (feature_max - feature_min)
        
        test_batch = torch.from_numpy(test_batch).float().cuda()
        test_label = torch.from_numpy(test_label).long().cuda()
        
        logits = model(test_batch)
        
        _, predictions = torch.max(logits, dim=1)
        test_acc_num += torch.sum(predictions == test_label).item()
        
        idx += batch_size

    return test_acc_num / X_test.shape[0] * 100


class CrossEntropyLabelSmooth(nn.Module):
    """Cross Entropy loss with label smoothing for improved generalization."""
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def set_global_seed(seed):
    """Set the seed for random number generators to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def sort_a_b(a, b):
    """Sort `a` and `b` in the same order."""
    combined = list(zip(a, b))
    sorted_combined = sorted(combined)
    a_sorted, b_sorted = zip(*sorted_combined)
    return a_sorted, b_sorted


def set_prioity(cls_fix):
    """Set priority for rules based on class fixation."""
    priority_dict = {}
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        if len(set([int(i)//10000 for i in ids])) == 1:  # Skip overlapping prototypes
            continue

        for id in ids:
            if id not in priority_dict:
                priority_dict[id] = 1

            if int(id)//10000 == value:
                priority_dict[id] += 1

    # Sort priorities in ascending order
    priority_dict = sorted(priority_dict.items(), key=lambda x: (x[1], int(x[0])))
    priority_dict = {item[0]: index + 1 for index, item in enumerate(priority_dict)}

    add_class_rules_keys = []
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        if len(set([int(i)//10000 for i in ids])) == 1:
            continue

        priori_list = [priority_dict[i] for i in ids]
        prior_id = ids[np.argmax(priori_list)]
        if int(prior_id)//10000 != value:
            add_class_rules_keys.append(key)

    return priority_dict, add_class_rules_keys


def check_rule(sample, thres_list):
    """Check if a sample satisfies all the threshold conditions."""
    for dim1 in range(sample.shape[0]):
        if sample[dim1] < thres_list[dim1][0] or sample[dim1] > thres_list[dim1][1]:
            return False
    return True

# Function to determine if a directed path exists in a graph
def have_path(restrict, st, ed):
    """Check if a directed path exists between two points."""
    graph = {}
    for u, v in restrict:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
    
    # Depth-First Search (DFS) to check path existence
    def dfs(node, target, visited):
        if node == target:
            return True
        visited.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor, target, visited):
                    return True
        return False

    visited = set()
    return dfs(st, ed, visited)


def cal_rule_num(cls_fix, num_classes, class_rules): 
    """
    Calculate the number of rules and constraints for a given set of class fixation and class rules.
    This function also handles the process of adding constraints based on the prototypes and their categories.
    """
    res = 0  # Counter for the number of rules that need to be adjusted
    cnt = 0  # Counter for the total number of iterations (used for debugging or analysis)

    # Define constraints to be added based on the rules
    restrict = []
    reduced_cls_fix = {}

    for item in cls_fix:
        gt = cls_fix[item]  # Ground truth for the current class fixation
        proto_list = [s for s in item.split('#') if s]  # Filter out empty strings from the list
        for idx in range(len(proto_list)):
            proto_list[idx] = int(proto_list[idx])  # Convert all elements to integers
            
        # Check if all prototypes belong to the same category
        flag = 1  # Flag to check if all prototypes are from the same category
        for idx in range(len(proto_list)):
            if proto_list[idx] // 10000 != proto_list[0] // 10000:
                flag = 0  # If categories differ, set flag to 0
                break
        
        # If all prototypes belong to the same category
        if flag == 1:
            if gt != proto_list[0] // 10000:  # If the ground truth doesn't match the prototype's category
                reduced_cls_fix[item] = gt
                res += 1  # Increment the result counter
            continue
        
        # If the ground truth category does not exist among the prototypes, simply add the rule
        flag = 1
        for idx in range(len(proto_list)):
            if gt == proto_list[idx] // 10000:  # Check if ground truth matches any prototype
                flag = 0
                break
        if flag == 1:
            reduced_cls_fix[item] = gt
            res += 1  # Increment the result counter
            continue
        
        # Begin adding constraints (similar to topological sorting)
        gt_proto = []  # List to store prototypes belonging to the ground truth category
        other_proto = []  # List to store other prototypes not belonging to the ground truth category
        
        for idx in range(len(proto_list)):
            if proto_list[idx] // 10000 == gt:
                gt_proto.append(proto_list[idx])  # Add ground truth prototypes
            else:
                other_proto.append(proto_list[idx])  # Add other prototypes
        
        # Attempt to add constraints based on prototype relations
        flag = 1
        add_edge_num = 0  # Counter for the number of edges (constraints) added
        for dim1 in range(len(gt_proto)):
            if flag == 0:
                break  # Break if a valid condition is found
            for dim2 in range(len(other_proto)):
                cnt += 1  # Increment the iteration counter
                # Check if there is an existing directed path from `other_proto` to `gt_proto`
                if have_path(restrict, other_proto[dim2], gt_proto[dim1]):
                    flag = 0  # If a path exists, the constraint is not valid
                    break
                else:
                    add_edge_num += 1
                    restrict.append((gt_proto[dim1], other_proto[dim2]))  # Add the new constraint (edge)
        
        # If the constraint can't be satisfied, remove the added edges
        if flag == 0:
            reduced_cls_fix[item] = gt
            res += 1
            # Rollback by removing the edges added during the process
            for _ in range(add_edge_num):
                restrict.pop()

    return res, reduced_cls_fix  # Return the number of rules adjusted and the reduced class fixation


# If more than one table entry matches successfully, the result of the table entry with the highest priority is returned (this is also true on the switch)
def solve_conflict(old_X, old_y,num_classes,class_rules,topo_sort = False):
    
 
    cls_fix = {}
    
    for idx in range(old_X.shape[0]):
        
        sample = old_X[idx]
        cls_res = []
        
        for dim1 in range(num_classes):
            for dim2 in range(len(class_rules[dim1])):
                if check_rule(sample, class_rules[dim1][dim2]):
                    
                    if (dim1 * 10000 + dim2) not in cls_res:
                        cls_res.append(dim1 * 10000 + dim2)
        
        if len(cls_res) <= 1:
            continue
        
        # cls_res has more than 1 class
        cls_res.sort(reverse=False)
        cls_res_hash = '#'
        for p in cls_res:
            cls_res_hash += str(p)
            cls_res_hash += '#'
        
        if cls_res_hash not in cls_fix:
            cls_fix[cls_res_hash] = []

        cls_fix[cls_res_hash].append(old_y[idx])
    
    cls_fix = dict(sorted(cls_fix.items(), key=lambda item: Counter(item[1]).most_common(1)[0][1], reverse=True))

    for p in cls_fix:
        cls_fix[p] = statistics.mode(cls_fix[p])
    
    
    
    # confilct rule num
    if topo_sort==True:
        conflict_rule_num,reduced_cls_fix = cal_rule_num(cls_fix,num_classes,class_rules)

        return cls_fix,reduced_cls_fix,conflict_rule_num
    else:
        return cls_fix,0,0



def intersect_rules(list1, list2):
    # Create an empty list to store the intersected rules
    intersected_rules = []

    # Ensure both lists have the same length to compare corresponding elements
    if len(list1) != len(list2):
        raise ValueError("The lengths of the two lists must be the same")

    # Iterate through both lists and compute the intersection of corresponding rules
    for i in range(len(list1)):
        lower_bound = max(list1[i][0], list2[i][0])  # Take the larger lower bound
        upper_bound = min(list1[i][1], list2[i][1])  # Take the smaller upper bound
        
        # If the lower bound is less than or equal to the upper bound, there's an intersection
        if lower_bound <= upper_bound:
            intersected_rules.append([lower_bound, upper_bound])
        else:
            return -1  # Return -1 if there is no intersection for this rule
    
    return intersected_rules  # Return the list of intersected rules


def convert_bounds_to_integers(rules):
    new_rules = copy.deepcopy(rules)
    # 遍历规则列表的每一个元素
    for category in range(len(new_rules)):
        for rule in range(len(new_rules[category])):
            for feature in range(len(new_rules[category][rule])):
                # 将上下界转换为整数
                new_rules[category][rule][feature][0] = int(new_rules[category][rule][feature][0])  # 下界
                new_rules[category][rule][feature][1] = int(new_rules[category][rule][feature][1])  # 上界
    return new_rules


def conflict_2rule(cls_fix, class_rules, num_classes):
    """Identify conflicts between rules based on class fixation and original class rules."""
    conflict_rules = [[] for i in range(num_classes)]  # Initialize list to store conflict rules for each class
    conflict_rules_pri = [[] for i in range(num_classes)]  # Initialize list to store priorities for conflict rules

    # Iterate over the class fixation dictionary to find conflicts
    for key, value in cls_fix.items():
        ids = key.strip('#').split('#')
        
        for i, id in enumerate(ids):
            dim1 = int(id) // 10000  # Extract the class dimension from the id
            dim2 = int(id) % 10000  # Extract the prototype dimension from the id
            
            if i == 0:
                tmp = copy.deepcopy(class_rules[dim1][dim2])  # Get the rule for the first prototype
            else:
                tmp2 = copy.deepcopy(class_rules[dim1][dim2])  # Get the rule for subsequent prototypes
                tmp = intersect_rules(tmp, tmp2)  # Find the intersection of the two rules
                
                if tmp == -1:
                    break  # Break if there is no valid intersection
        if tmp == -1:
            continue  # Skip this conflict if no intersection was found
        
        # Add the resulting rule to the conflict rules
        conflict_rules_pri[int(value)].append(10000 + len(ids))
        conflict_rules[int(value)].append(tmp)

    return conflict_rules, conflict_rules_pri


def add_rules(args, dim1, param_names, param_info, f, class_rules):
    """Add new rules to a rule set based on the class rules and parameters."""
    s = 0  # Counter for added rules
    for dim2 in range(len(class_rules[dim1])):
        params = []
        src_port_value = 0
        src_port_mask = 0
        dst_port_value = 0
        dst_port_mask = 0

        # Process each parameter in the rule and generate the corresponding action
        for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
            if start < 0:
                start = 0  # Ensure the start is non-negative
            if end > param_info[param_name]:
                end = param_info[param_name]  # Ensure the end doesn't exceed the parameter's maximum value

            # Handle srcPort_0 to srcPort_15
            if 'srcport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # Extract bit position from the parameter name
                if start == 0 and end == 1:
                    src_port_mask &= ~(1 << bit_position)  # Mask the bit position
                elif start == 0 and end == 0:
                    src_port_value &= ~(1 << bit_position)  # Set value to 0 and mask the bit
                    src_port_mask |= (1 << bit_position)  # Mask the bit
                elif start == 1 and end == 1:
                    src_port_value |= (1 << bit_position)  # Set value to 1 and mask the bit
                    src_port_mask |= (1 << bit_position)  # Mask the bit

            # Handle dstPort_0 to dstPort_15
            elif 'dstport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # Extract bit position
                if start == 0 and end == 1:
                    dst_port_mask &= ~(1 << bit_position)  # Mask the bit position
                elif start == 0 and end == 0:
                    dst_port_value &= ~(1 << bit_position)  # Set value to 0 and mask the bit
                    dst_port_mask |= (1 << bit_position)  # Mask the bit
                elif start == 1 and end == 1:
                    dst_port_value |= (1 << bit_position)  # Set value to 1 and mask the bit
                    dst_port_mask |= (1 << bit_position)  # Mask the bit

            else:
                # For other parameters, simply append the start and end values
                params.append(f"{param_name}_start={start}, {param_name}_end={end}")

        # If dataset is 'unsw-nb15', include the srcPort and dstPort information in the action
        if args.dataset == "unsw-nb15":
            params.append(f"meta_srcport={src_port_value}, meta_srcport_mask={src_port_mask}")
            params.append(f"meta_dstport={dst_port_value}, meta_dstport_mask={dst_port_mask}")

        # Add the port category
        params.append(f"port={dim1}")
        params_str = ", ".join(params)  # Convert list of parameters to a string

        # Write the rule generation action to the file
        f.write(f"tb_packet_cls.add_with_ac_packet_forward({params_str})\n")
        s += 1  # Increment the counter

    return s


def delete_rules(args, dim1, param_names, param_info, f, class_rules):
    """Delete existing rules from a rule set based on the class rules and parameters."""
    s = 0  # Counter for deleted rules
    for dim2 in range(len(class_rules[dim1])):
        params = []
        src_port_value = 0
        src_port_mask = 0
        dst_port_value = 0
        dst_port_mask = 0

        # Process each parameter in the rule and generate the corresponding action
        for param_name, (start, end) in zip(param_names, class_rules[dim1][dim2]):
            if start < 0:
                start = 0
            if end > param_info[param_name]:
                end = param_info[param_name]

            # Handle srcPort_0 to srcPort_15
            if 'srcport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # Extract bit position from the parameter name
                if start == 0 and end == 1:
                    src_port_mask &= ~(1 << bit_position)  # Mask the bit position
                elif start == 0 and end == 0:
                    src_port_value &= ~(1 << bit_position)  # Set value to 0 and mask the bit
                    src_port_mask |= (1 << bit_position)  # Mask the bit
                elif start == 1 and end == 1:
                    src_port_value |= (1 << bit_position)  # Set value to 1 and mask the bit
                    src_port_mask |= (1 << bit_position)  # Mask the bit

            # Handle dstPort_0 to dstPort_15
            elif 'dstport' in param_name:
                bit_position = int(param_name.split('_')[-1])  # Extract bit position
                if start == 0 and end == 1:
                    dst_port_mask &= ~(1 << bit_position)  # Mask the bit position
                elif start == 0 and end == 0:
                    dst_port_value &= ~(1 << bit_position)  # Set value to 0 and mask the bit
                    dst_port_mask |= (1 << bit_position)  # Mask the bit
                elif start == 1 and end == 1:
                    dst_port_value |= (1 << bit_position)  # Set value to 1 and mask the bit
                    dst_port_mask |= (1 << bit_position)  # Mask the bit

            else:
                # For other parameters, append start and end values to the parameters list
                params.append(f"{param_name}_start={start}, {param_name}_end={end}")

        # If dataset is 'unsw-nb15', include the srcPort and dstPort information in the action
        if args.dataset == "unsw-nb15":
            params.append(f"meta_srcport={src_port_value}, meta_srcport_mask={src_port_mask}")
            params.append(f"meta_dstport={dst_port_value}, meta_dstport_mask={dst_port_mask}")

        # Convert the parameters list into a string
        params_str = ", ".join(params)
        f.write(f"tb_packet_cls.delete({params_str})\n")  # Write the delete rule action to the file
        s += 1  # Increment the counter

    return s
