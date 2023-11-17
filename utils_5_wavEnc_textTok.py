# -*- coding: utf-8 -*-
# @Time : 2023/10/18 13:53
# @Author : JohnnyYuan
# @File : utils_5_wavEncoded_text.py

import os
import time
import random
import argparse
import pickle
import copy
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.utils.rnn as rmm_utils
import torch.utils.data.dataset as Dataset
from sklearn import preprocessing

import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


class subDataset(Dataset.Dataset):
    def __init__(self, Data_1, Label,ids , att):
        self.Data_1 = Data_1
        self.Label = Label
        self.Ids = ids
        self.Att = att

    def __len__(self):
        return len(self.Data_1)

    def __getitem__(self, item):
        data_1 = torch.Tensor(self.Data_1[item])
        label = torch.Tensor(self.Label[item])
        ids_1 = torch.Tensor(self.Ids[item])
        att_1 = torch.Tensor(self.Att[item])
        return data_1, label ,ids_1, att_1


def STD(input_fea):
    data_1 = []
    for i in range(len(input_fea)):
        data_1.append(input_fea[i])
    a = []
    for i in range(len(data_1)):
        a.extend(data_1[i])
    scaler_1 = preprocessing.StandardScaler().fit(a)
    # print(scaler_1.mean_)
    # print(scaler_1.var_)
    for i in range(len(input_fea)):
        input_fea[i] = scaler_1.transform(input_fea[i])
    return input_fea


def Feature(data, args):
    input_data_spec = []
    # 按照遍历顺序拿取数据
    for i in range(len(data)):
        input_data_spec.append(data[i]['wav_encodings'])  # 此处拿取的是经过wav2vec2后的数据

    '''
    a = [0.0 for i in range(args.utt_insize)]
    a = np.array(a)
    input_data_spec_CNN = []

    for i in range(len(input_data_spec)):
        ha = []
        if(len(input_data_spec[i]) < 300):
            for z in range(len(input_data_spec[i])):
                ha.append(np.array(input_data_spec[i][z]))
            len_zero = 300 - len(input_data_spec[i])
            for x in range(len_zero):
                ha.append(np.array(a))
        if(len(input_data_spec[i]) >= 300):
            for z in range(len(input_data_spec[i])):
                if(z < 300):
                    ha.append(np.array(input_data_spec[i][z]))
        ha = np.array(ha)
        input_data_spec_CNN.append(ha)
    input_data_spec_CNN = STD(input_data_spec_CNN)
    '''

    input_label = []
    for i in range(len(data)):
        input_label.append(data[i]['label'])
    input_data_id = []
    for i in range(len(data)):
        input_data_id.append(data[i]['id'])
    input_label_org = []
    for i in range(len(data)):
        input_label_org.append(data[i]['label'])
    input_ids = []
    for i in range(len(data)):
        input_ids.append(data[i]['input_ids'])
    input_att_mask = []
    for i in range(len(data)):
        input_att_mask.append(data[i]['attention_mask'])
    return input_data_spec, input_data_spec, input_label, input_data_id, input_label_org ,input_ids, input_att_mask


def Get_data(data, train, test, args):  # test 从[0,1,2,3,4]里抽一个，train则取剩下的，都是列表格式
    train_data = []
    test_data = []
    for i in range(len(train)):
        train_data.extend(data[train[i]])  # 把目标组存入train_data [s1,s2,s3,s4],组别之间不再分开
    for i in range(len(test)):
        test_data.extend(data[test[i]])  # 把目标组存入test_data

    input_train_data_spec, input_train_data_spec_CNN, input_train_label, input_train_data_id, _ ,input_train_ids, input_train_att= Feature(train_data, args)
    input_test_data_spec, input_test_data_spec_CNN, input_test_label, input_test_data_id, input_test_label_org, input_test_ids, input_test_att= Feature(test_data, args)

    # label = np.array(input_train_label, dype='int64').reshape(-1,1)
    label = np.array(input_train_label).reshape(-1, 1)
    label_test = np.array(input_test_label).reshape(-1, 1)
    train_dataset = subDataset(input_train_data_spec_CNN, label, input_train_ids, input_train_att)
    test_dataset = subDataset(input_test_data_spec_CNN, label_test, input_test_ids, input_test_att)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False)
    return train_loader, test_loader, input_test_data_id, input_test_label_org