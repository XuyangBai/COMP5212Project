import numpy as np
from sklearn.metrics import hamming_loss
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable

'''
Helper methods for evaluating a classification network
(hamming loss,  Jaccard index, precision, recall, f1_score)
The output of the model and test_label should be in the format of [batch_size , n_category] np.matrix
'''

class evaluator:
    def __init__(self, trained_model, test_img, test_label):
        self.trained_model = trained_model
        self.test_img = test_img
        self.test_label = test_label
        self.hamming_loss = 0
        self.Jaccard_index = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
    
    def __count_hamming_loss(self, outputs, targets):
        hl_sum = 0
        for i in outputs.shape[0]:
            hl_sum += hamming_loss(targets[i], outputs[i])
        self.hamming_loss = hl_sum / outputs.shape[0]
    
    def __cout_Jaccard_index(self, outputs, targets):
        Ji_sum = 0
        for i in outputs.shape[0]:
            Ji_sum += jaccard_similarity_score(targets[i], outputs[i])
        self.Jaccard_index = Ji_sum / outputs.shape[0]

    def __count_precision_and_recall(self, outputs, targets):
        pre_sum = 0
        rec_sum = 0
        for i in outputs.shape[0]:
            cm = confusion_matrix(targets[i], outputs[i])
            rec_sum += np.diag(cm) / np.sum(cm, axis = 1)
            pre_sum += np.diag(cm) / np.sum(cm, axis = 0)
        self.recall = rec_sum / outputs.shape[0]
        self.precision = pre_sum / outputs.shape[0]

    def __count_f1_score(self):
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)


    def __forward_pass(self):
        ''' forward in_ through the net, return outputs '''
        ''' forward process of the trained model is defined as net_forward '''
        input_var = Variable(self.test_img).cuda(async=True)
        outputs = self.trained_model.net_forward(input_var)
        return outputs

    def evaluate(self):
        ''' evaluate the net on the data in the loader '''
        outputs = self.__forward_pass()
        self.__count_hamming_loss(outputs, self.test_label)
        self.__cout_Jaccard_index(outputs, self.test_label)
        self.__count_precision_and_recall(outputs, self.test_label)
        self.__count_f1_score()
    
    def get_hamming_loss(self):
        return self.hamming_loss
    
    def get_Jaccard_index(self):
        return self.Jaccard_index
    
    def get_precision(self):
        return self.precision

    def get_recall(self):
        return self.recall
    
    def get_f1_score(self):
        return self.f1_score