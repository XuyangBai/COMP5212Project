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


def __count_hamming_loss(outputs, targets):
    hl_sum = 0
    for i in outputs.shape[0]:
        hl_sum += hamming_loss(targets[i], outputs[i])
    return hl_sum / outputs.shape[0]


def __cout_Jaccard_index(outputs, targets):
    Ji_sum = 0
    for i in outputs.shape[0]:
        Ji_sum += jaccard_similarity_score(targets[i], outputs[i])
    return Ji_sum / outputs.shape[0]


# def __count_precision_and_recall(outputs, targets):
#     pre_sum = 0
#     rec_sum = 0
#     for i in outputs.shape[0]:
#         cm = confusion_matrix(targets[i], outputs[i])
#         rec_sum += np.diag(cm) / np.sum(cm, axis=1)
#         pre_sum += np.diag(cm) / np.sum(cm, axis=0)
#     recall = rec_sum / outputs.shape[0]
#     precision = pre_sum / outputs.shape[0]
#     return recall, precision
#
# def __count_f1_score(self):
#     self.f1_score = 2 * (precision * self.recall) / (self.precision + self.recall)

def __count_acc(outputs, targets):
    # [n,1,28]
    acc = [0 for _ in range(28)]
    for i in range(28):
        acc[i] = (outputs[:, i] == targets[:, i]).sum() / outputs.shape[0]
    return acc


def __forward_pass(model, input):
    ''' forward in_ through the net, return outputs '''
    ''' forward process of the trained model is defined as net_forward '''
    outputs = model(input)
    return outputs


def evaluate(trained_model, loader, device):
    ''' evaluate the net on the data in the loader '''
    model = trained_model
    hamming_loss = []
    Jaccard_index = []
    accuracy = [[] for _ in range(28)]
    for i_batch, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.float()
        labels.squeeze_(dim=1)
        outputs = __forward_pass(model, images)
        batch_acc = __count_acc(outputs, labels)
        for i in range(28):
            accuracy[i].append(batch_acc[i])
#        hamming_loss.append(__count_hamming_loss(outputs, labels))
#        Jaccard_index.append(__cout_Jaccard_index(outputs, labels))
    # calculate average accuracy for each class
    for i in range(28):
        accuracy[i] = np.mean(accuracy[i])
    metrix = {
#        'hamming': np.mean(hamming_loss),
#        'jaccard': np.mean(Jaccard_index),
        'accuracy': np.array(accuracy).mean(),
    }
    return metrix