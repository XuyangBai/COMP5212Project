import numpy as np

import torch
from torch.autograd import Variable
from dataloader import get_data_loader

'''
Helper methods for evaluating a classification network
(hamming loss,  Jaccard index, precision, recall, f1_score)
The output of the model and test_label should be in the format of [batch_size , n_category] np.matrix
'''


# Not used in this project
# def __count_hamming_loss(outputs, targets):
#     hl_sum = 0
#     for i in outputs.shape[0]:
#         hl_sum += hamming_loss(targets[i], output
# s[i])
#     return hl_sum / outputs.shape[0]


# def __cout_Jaccard_index(outputs, targets):
#     Ji_sum = 0
#     for i in outputs.shape[0]:
#         Ji_sum += jaccard_similarity_score(targets[i], outputs[i])
#     return Ji_sum / outputs.shape[0]


def __count_TP_P_T(outputs, targets):
    tp_sum = []
    p_sum = []
    t_sum = []
    for i in range(28):
        tp = ((outputs[:, i] == targets[:, i]) * targets[:, i]).sum()
        p = outputs[:, i].sum()
        t = targets[:, i].sum()
        tp_sum.append(tp)
        p_sum.append(p)
        t_sum.append(t)
    metric = {
        'tp_sum' : tp_sum,
        'p_sum' : p_sum,
        't_sum' : t_sum
    }
    return metric
    

def __count_acc(outputs, targets):
    # [n,1,28]
    acc = [0 for _ in range(28)]
    for i in range(28):
        acc[i] = (outputs[:, i] == targets[:, i]).float().sum() / outputs.shape[0]
    return acc


def __forward_pass(model, input):
    ''' forward input through the net, return outputs '''
    outputs = model(input)
    return outputs


def evaluate(trained_model, loader, device):
    ''' evaluate the net on the data in the loader '''
    model = trained_model
    accuracies = [[] for _ in range(28)]
    TP_sum = []
    P_sum = []
    T_sum = []
    recalls = []
    precisions = []
    for i_batch, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.byte()
        images = images.float()
        outputs = __forward_pass(model, images) >= 0
        metric = __count_TP_P_T(outputs, labels)
        tp_sum = metric.get('tp_sum')
        p_sum = metric.get('p_sum')
        t_sum = metric.get('t_sum')
        TP_sum.append(tp_sum)
        P_sum.append(p_sum)
        T_sum.append(t_sum)
        batch_acc = __count_acc(outputs, labels)
        for i in range(28):
            accuracies[i].append(batch_acc[i])
    # calculate average accuracy for each class
    for i in range(28):
        accuracies[i] = np.mean(accuracies[i])

    # calculate recall, precision and f1_score over all the test set
#    TP = [sum(x) for x in zip(*TP_sum)]
#    P = [sum(x) for x in zip(*P_sum)]
#    T = [sum(x) for x in zip(*T_sum)]
    TP = [0 for _ in range(28)]
    T = [0 for _ in range(28)]
    P = [0 for _ in range(28)]
    for i in range(28):
        TP[i] = sum([x[i] for x in TP_sum])
        T[i] = sum([x[i] for x in T_sum])
        P[i] = sum([x[i] for x in P_sum])
        # for each class calculate precision and recall
    for i in range(28):
        if P[i].float() == 0:
            precisions.append(P[i].float()) # append 0
        else:
            precisions.append(TP[i].float() / P[i].float())
        if T[i].float() == 0:
            recalls.append(T[i].float())
        else:
            recalls.append(TP[i].float() / T[i].float())
    # calculate the average precision and recall over all the classes
    accuracies = np.array(accuracies)
    precisions, recalls = np.array(precisions), np.array(recalls)
    precision = precisions.mean()
    recall = recalls.mean()
    f1_score = 2 * (recall * precision) / (recall + precision)
    metric = {
#        'hamming': np.mean(hamming_loss),
#        'jaccard': np.mean(Jaccard_index),
        'f1_macro': f1_score,
        'acc': accuracies.mean(),
        'prec': precision,
        'recl': recall,
        'acc_all': accuracies,
        'prec_all': precisions,
        'recl_all': recalls,
    }
    return metric




