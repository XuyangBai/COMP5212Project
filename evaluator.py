import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix
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
#         hl_sum += hamming_loss(targets[i], outputs[i])
#     return hl_sum / outputs.shape[0]


# def __cout_Jaccard_index(outputs, targets):
#     Ji_sum = 0
#     for i in outputs.shape[0]:
#         Ji_sum += jaccard_similarity_score(targets[i], outputs[i])
#     return Ji_sum / outputs.shape[0]


def __count_f1_score(outputs, targets):
    precisions = []
    recalls = []
    for i in range(28):
        TP = (outputs[:, i] == targets[:, i]).sum()
        P = outputs[:, i].sum()
        T = targets[:, i].sum()
        precisions.append(TP.float() / P.float())
        recalls.append(TP.float() / T.float())
    precision = np.array(precisions).mean()
    recall = np.array(recalls).mean()
    f1_score = 2 * (recall * precision) / (recall + precision)
    return f1_score
    

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
    accuracy = [[] for _ in range(28)]
    for i_batch, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.byte()
        labels.squeeze_(dim=1)
        outputs = __forward_pass(model, images) >= 0
        f1_score = __count_f1_score(outputs, labels)
        batch_acc = __count_acc(outputs, labels)
        for i in range(28):
            accuracy[i].append(batch_acc[i])
#        hamming_loss.append(__count_hamming_loss(outputs, labels))
#        Jaccard_index.append(__cout_Jaccard_index(outputs, labels))
    # calculate average accuracy for each class
    for i in range(28):
        accuracy[i] = np.mean(accuracy[i])
    metric = {
#        'hamming': np.mean(hamming_loss),
#        'jaccard': np.mean(Jaccard_index),
        'f1_score': f1_score,
        'accuracy': np.array(accuracy).mean(),
    }
    return metric