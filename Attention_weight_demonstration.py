# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:29:37 2023

@author: Iwasaki
"""

n_minibatch_size = 4
int_random_seed = 0
n_epoches = 20
int_which_cuda = 0
boolean_reccurent_layer = True
learning_rate = 0.0001
n_max_word = 20


import pandas as pd
df_dataset = pd.read_csv("G:\Google//NUS\PhD_Statistics_master\PhD_thesis\Toy_example\FinancialPhraseBank.csv")
df_dataset = df_dataset.sample(frac=1, random_state=0).reset_index(drop=True)

#################################
# Word2vec training 
#################################
l_l_words = []
for i_sample in range(len(df_dataset)):
    str_sentence = df_dataset.loc[i_sample, 'sentence']
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    l_words = tokenizer.tokenize(str_sentence) 
    l_words_lowered = [i.lower() for i in l_words]
    l_l_words.append(l_words_lowered)

df_dataset['l_word'] = l_l_words

from gensim.models import Word2Vec
gensim_word2vec = Word2Vec(l_l_words, sg =0, size=50, min_count= 4, window= 3, alpha = 0.025,workers = 4, negative = 5, iter = 5, sorted_vocab = 1)

word_embeddings_keyedvec = gensim_word2vec.wv
n_vocaburary = word_embeddings_keyedvec.vectors.shape[0]
d_embeddings = word_embeddings_keyedvec.vectors.shape[1]

l_vocabulary_word_index = []
dic_vocabulary_index_to_word = {}
l_word_vocabulary = list(word_embeddings_keyedvec.vocab.keys())
for i_word in range(len(l_word_vocabulary)):
    str_word = l_word_vocabulary[i_word]
    i_word_index = word_embeddings_keyedvec.vocab[str_word].index
    l_vocabulary_word_index.append(i_word_index)
    dic_vocabulary_index_to_word[i_word_index] = str_word
    
# Preparation for padding
import numpy as np
index_padding_position = n_vocaburary
v_zeros_padding = np.zeros((1, word_embeddings_keyedvec.vectors.shape[1]))
word2vec_with_zero_row = np.concatenate((word_embeddings_keyedvec.vectors, v_zeros_padding), axis=0) 
# The position at n_vocaburary has v_zeros_padding. # 3125

df_dataset['l_word_id'] = ''
df_dataset['l_word_in_vocabulary'] = ''
df_dataset['l_word_in_vocabulary_padded'] = ''
df_dataset['end_word_position'] = ''

for i_loop1 in range(len(df_dataset)):
    print('preprocessing:', i_loop1, 'out of', len(df_dataset))
    l_word_id = [] 
    l_word_occurance = []        
    l_word_temp = df_dataset['l_word'][i_loop1]  
    l_end_position_word_temp = []
    n_not_in_vocaburary = 0
    n_loop2 = min(len(l_word_temp), n_max_word)
    for i_loop2 in range(n_loop2):
        str_word_temp = l_word_temp[i_loop2]
        if str_word_temp in word_embeddings_keyedvec.vocab:
            int_key_temp = word_embeddings_keyedvec.vocab[str_word_temp].index  
            l_word_id.append(int_key_temp)
            l_word_occurance.append(str_word_temp)
        else:
            n_not_in_vocaburary += 1
    n_legitimate_word = len(l_word_id)
    n_paddings_word_level = n_max_word - n_legitimate_word
    l_word_id_padded = l_word_id + [index_padding_position]*n_paddings_word_level
    l_word_occurance_padded = l_word_occurance + ['']*n_paddings_word_level    
    df_dataset.at[i_loop1, 'l_word_id'] = l_word_id_padded
    df_dataset.at[i_loop1,'l_word_in_vocabulary'] = l_word_occurance
    df_dataset.at[i_loop1,'l_word_in_vocabulary_padded'] = l_word_occurance_padded
    df_dataset.at[i_loop1,'end_word_position'] = n_legitimate_word

df_dataset_train = df_dataset.loc[:199, :].reset_index(drop=True)
df_dataset_test = df_dataset.loc[199:, :].reset_index(drop=True)

##############################################
# Data Loader
##############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torch.utils as utils
from torchvision import datasets, transforms
import random

tt_X_keyvec_train = torch.Tensor(df_dataset_train['l_word_id']).long()
tt_X_keyvec_test = torch.Tensor(df_dataset_test['l_word_id']).long()

tt_X_keyvec_word_end_position_train = torch.Tensor(df_dataset_train['end_word_position']).long().clamp(min=1, max=n_max_word)
tt_X_keyvec_word_end_position_test = torch.Tensor(df_dataset_test['end_word_position']).long().clamp(min=1, max=n_max_word) # Control the maximum sentence length. 

tt_y_train = torch.Tensor(df_dataset_train['score']).float() 
tt_y_test = torch.Tensor(df_dataset_test['score']).float()

tt_dataset_train = torch.utils.data.TensorDataset(tt_X_keyvec_train, tt_X_keyvec_word_end_position_train, tt_y_train)
tt_dataset_test = torch.utils.data.TensorDataset(tt_X_keyvec_test, tt_X_keyvec_word_end_position_test, tt_y_test)

np.random.seed(0)
tt_data_loaded_train = utils.data.DataLoader(tt_dataset_train, batch_size = n_minibatch_size, shuffle=False, drop_last=True,  num_workers= 0) 
tt_data_loaded_test = utils.data.DataLoader(tt_dataset_test, batch_size = n_minibatch_size, shuffle=False, drop_last=True, num_workers= 0) 

tt_word2vec = torch.FloatTensor(word2vec_with_zero_row) 
word_embeddings_pretrained = nn.Embedding.from_pretrained(embeddings = tt_word2vec, freeze=False, padding_idx =index_padding_position, max_norm =None)
d_word_embeddings = word2vec_with_zero_row.shape[1]

##############################################
# utility functions 
##############################################
def element_wise_mul(input1, input2):
    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    output = torch.cat(feature_list, 0)
    return torch.sum(output, 0).unsqueeze(0)


from sklearn import metrics
from scipy.stats import pearsonr
def get_evaluation(y_true, y_prob, list_metrics):
    y_pred = np.argmax(y_prob, -1)
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    if 'correlation' in list_metrics:
        try:
            output['correlation'] = pearsonr(y_true, y_prob.reshape(-1))[0]
        except ValueError:
            output['correlation'] = -1
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


###############################################
# HAN
###############################################
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import csv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
d_word_hidden = 32

class HAN(nn.Module):
    def __init__(self, word_embeddings_pretrained, word_hidden_size, batch_size, num_classes):
        super(HAN, self).__init__()
        self.lookup = word_embeddings_pretrained
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size 
        self.lstm = nn.LSTM(input_size = d_word_embeddings, hidden_size = word_hidden_size, bidirectional=True)
        self.word_fc = nn.Linear(2*word_hidden_size, 2*word_hidden_size)
        self.context_weight = nn.Parameter(torch.randn(2 * word_hidden_size, 1), requires_grad = True)
        self.final_fc = nn.Linear(word_hidden_size*2, num_classes)         
        self._init_hidden_state()

    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.word_cell_state = torch.zeros(2, batch_size, self.word_hidden_size)
         
        if torch.cuda.is_available():
            self.word_hidden_state = self.word_hidden_state.cuda(int_which_cuda)
            self.word_cell_state = self.word_cell_state.cuda(int_which_cuda)
            
    def forward(self, tt_input_ids, tt_end_word_position):
        tt_input_ids_permuted = tt_input_ids.permute(1,0) 
        n_max_length = tt_input_ids_permuted.shape[0]
        tt_embeddings = self.lookup(tt_input_ids_permuted)
        input_packed = pack_padded_sequence(input=tt_embeddings, lengths=tt_end_word_position, batch_first=False, enforce_sorted=False) 
        f_output, (h_final, c_final) = self.lstm(input_packed, (self.word_hidden_state, self.word_cell_state))
        f_output, end_word_position_by_batch = pad_packed_sequence(f_output, batch_first=False, total_length=n_max_length)        

        u_context = self.word_fc(f_output.permute(1, 0, 2)) 
        u_context = torch.tanh(u_context)
        u_context = u_context.permute(1, 0, 2) 

        mask_for_u_context = torch.ones(u_context.shape[0], u_context.shape[1], u_context.shape[2], dtype=torch.float)
        for i_mask_batch in range(len(tt_end_word_position)):
            mask_for_u_context[tt_end_word_position[i_mask_batch]:, i_mask_batch, :] = torch.tensor(0)
        if torch.cuda.is_available():
            mask_for_u_context = mask_for_u_context.cuda(int_which_cuda)         
        u_context = u_context*mask_for_u_context

        mask_for_softmax_word = torch.zeros(len(tt_end_word_position), n_max_word, dtype=torch.float)
        for i_mask_batch in range(len(tt_end_word_position)):
            mask_for_softmax_word[i_mask_batch, tt_end_word_position[i_mask_batch]:] = torch.tensor(float('-inf'))
        if torch.cuda.is_available():
            mask_for_softmax_word = mask_for_softmax_word.cuda(int_which_cuda)
        
        l_feature_list = []
        for i_batch in range(len(tt_end_word_position)):
            tt_feature_temp = torch.mm(u_context[:, i_batch, :].squeeze(1), self.context_weight) 
            l_feature_list.append(tt_feature_temp)
        a_attention_words = torch.stack(l_feature_list, axis=0)
        a_attention_words = a_attention_words.squeeze(2) 
        a_attention_words = a_attention_words + mask_for_softmax_word
        a_attention_words = F.softmax(a_attention_words, dim=1)
        v_output_word_net = element_wise_mul(f_output, a_attention_words.permute(1, 0))
        
        prediction = self.final_fc(v_output_word_net)
        return prediction, a_attention_words

###########################################
# train 
###########################################
l_l_df_true_hat_comparison_train = []
l_l_df_true_hat_comparison_eval = []

torch.manual_seed(int_random_seed)
n_target_classes = 1
HAN_model = HAN(word_embeddings_pretrained, d_word_hidden, n_minibatch_size, n_target_classes)

if torch.cuda.is_available():
    HAN_model.cuda(int_which_cuda)

criterion = torch.nn.MSELoss(reduction='mean')     
momentum = 0.9
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, HAN_model.parameters()), lr=learning_rate) 
num_iter_per_epoch = len(tt_data_loaded_train)

l_correlation_train = []
l_correlation_test = []
l_mse_train = []
l_mse_test = []
l_dic_y_hat_train = []
l_dic_y_hat_test = []
l_dic_y_true_train = []
l_dic_y_true_test = []

l_dic_word_attention_train = []
l_dic_word_attention_test = []

for epoch in range(n_epoches):
    l_y_hat_train = []
    l_y_true_train = []
    l_y_hat_test = []
    l_y_true_test = []

    # Train with optimizer.step ######################
    HAN_model.train()
    for i_minibatch, (tt_input_ids_train, tt_end_word_position_train, tt_y_train) in enumerate(tt_data_loaded_train):
        optimizer.zero_grad()
        if torch.cuda.is_available():
            tt_input_ids_train = tt_input_ids_train.cuda(int_which_cuda)
            tt_y_train = tt_y_train.cuda(int_which_cuda)
        HAN_model._init_hidden_state() 
        (outputs, l_attention) = HAN_model.forward(tt_input_ids_train, tt_end_word_position_train)
        loss = torch.mean((outputs.reshape(-1).squeeze(0) - tt_y_train)**2)
        loss.backward()
        optimizer.step()

    # Train last run ######################
    HAN_model.eval()
    l_y_pred_train = []
    l_y_true_train = []
    dic_word_attention_train = {}
    for i_word in range(len(l_vocabulary_word_index)):
        i_word_index = l_vocabulary_word_index[i_word]
        dic_word_attention_train[i_word_index] = []
    with torch.no_grad():
        for i_minibatch, (tt_input_ids_train, tt_end_word_position_train, tt_y_train) in enumerate(tt_data_loaded_train):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                tt_input_ids_train = tt_input_ids_train.cuda(int_which_cuda)
                tt_y_train = tt_y_train.cuda(int_which_cuda)
            HAN_model._init_hidden_state() 
            (outputs, tt_attention) = HAN_model.forward(tt_input_ids_train, tt_end_word_position_train)
            l_y_pred_train.extend(outputs.reshape(-1).cpu().detach().numpy())
            l_y_true_train.extend(tt_y_train.cpu().detach().numpy())            
            m_attention = tt_attention.cpu().detach().numpy()
            m_input_ids =tt_input_ids_train.cpu().detach().numpy()            
            for i_minibatch in range(len(tt_input_ids_train)):
                for i_word in range(m_input_ids.shape[1]):
                    rn_n_effective_word = sum(m_input_ids[i_minibatch, :] != index_padding_position)
                    if m_input_ids[i_minibatch, i_word] != index_padding_position:
                        i_word_id = m_input_ids[i_minibatch, i_word]
                        rn_attention_temp = m_attention[i_minibatch, i_word]
                        dic_word_attention_train[i_word_id].append(rn_attention_temp - 1/rn_n_effective_word)
    
    rn_mse_train = np.mean((np.array(l_y_true_train)-np.array(l_y_pred_train))**2)
    from scipy.stats import pearsonr
    rn_corrrelation_train = pearsonr(l_y_pred_train, l_y_true_train)[0]
    l_mse_train.append(rn_mse_train)
    l_correlation_train.append(rn_corrrelation_train)
    print('Epoch', epoch, 'train mse:', rn_mse_train)
    print('Epoch', epoch, 'train correlation:', rn_corrrelation_train)
    l_dic_word_attention_train.append(dic_word_attention_train)
        
    # Test last run ######################
    HAN_model.eval()
    l_y_pred_test = []
    l_y_true_test = []
    dic_word_attention_test = {}
    for i_word in range(len(l_vocabulary_word_index)):
        i_word_index = l_vocabulary_word_index[i_word]
        dic_word_attention_test[i_word_index] = []
    with torch.no_grad():
        for i_minibatch, (tt_input_ids_test, tt_end_word_position_test, tt_y_test) in enumerate(tt_data_loaded_test):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                tt_input_ids_test = tt_input_ids_test.cuda(int_which_cuda)
                tt_y_test = tt_y_test.cuda(int_which_cuda)
            HAN_model._init_hidden_state() 
            (outputs, tt_attention) = HAN_model.forward(tt_input_ids_test, tt_end_word_position_test)
            l_y_pred_test.extend(outputs.reshape(-1).cpu().detach().numpy())
            l_y_true_test.extend(tt_y_test.cpu().detach().numpy())            
            m_attention = tt_attention.cpu().detach().numpy()
            m_input_ids =tt_input_ids_test.cpu().detach().numpy()            
            for i_minibatch in range(len(tt_input_ids_test)):
                for i_word in range(m_input_ids.shape[1]):
                    rn_n_effective_word = sum(m_input_ids[i_minibatch, :] != index_padding_position)
                    if m_input_ids[i_minibatch, i_word] != index_padding_position:
                        i_word_id = m_input_ids[i_minibatch, i_word]
                        rn_attention_temp = m_attention[i_minibatch, i_word]
                        dic_word_attention_test[i_word_id].append(rn_attention_temp - 1/rn_n_effective_word)
    
    rn_mse_test = np.mean((np.array(l_y_true_test)-np.array(l_y_pred_test))**2)
    from scipy.stats import pearsonr
    rn_corrrelation_test = pearsonr(l_y_pred_test, l_y_true_test)[0]
    l_mse_test.append(rn_mse_test)
    l_correlation_test.append(rn_corrrelation_test)
    print('Epoch', epoch, 'test mse:', rn_mse_test)
    print('Epoch', epoch, 'test correlation:', rn_corrrelation_test)
    l_dic_word_attention_test.append(dic_word_attention_test)


from matplotlib import pyplot as plt
plt.plot(l_mse_train)
plt.plot(l_mse_test)
plt.plot(l_correlation_train)
plt.plot(l_correlation_test)

int_which_epoch = 10
dic_word_attention_recovered = l_dic_word_attention_train[int_which_epoch]

l_df_word_attention_summary = []
for i_word in range(len(l_word_vocabulary)):
    str_word = l_word_vocabulary[i_word]
    i_word_index = word_embeddings_keyedvec.vocab[str_word].index
    try:
        l_word_attention_observed = dic_word_attention_recovered[i_word_index]
    except:
        continue
    if len(l_word_attention_observed) < 3:
        continue
    rn_mean_excess_attention = np.mean(l_word_attention_observed)
    df_word_attention_temp = pd.DataFrame({'word':str_word, 'excess_attention':rn_mean_excess_attention}, index=[i_word])
    l_df_word_attention_summary.append(df_word_attention_temp)

df_word_attention_summary = pd.concat(l_df_word_attention_summary)

df_word_attention_summary = df_word_attention_summary.sort_values(by='excess_attention', ascending=False).reset_index(drop=True)
df_word_attention_summary = df_word_attention_summary.loc[df_word_attention_summary['word'].apply(lambda x:len(x)) > 1, :].reset_index(drop=True)

# top 10 attended words
print(df_word_attention_summary.head(10))

#         word  excess_attention
# 0        net          0.390077
# 1     profit          0.259981
# 2      sales          0.185049
# 3  operating          0.169306
# 4        eur          0.153330
# 5       loss          0.104787
# 6         up          0.087095
# 7       from          0.076385
# 8    quarter          0.057741
# 9       euro          0.051478


