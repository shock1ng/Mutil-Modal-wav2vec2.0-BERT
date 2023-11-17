# -*- coding: utf-8 -*-
# @Time : 2023/10/18 21:37
# @Author : JohnnyYuan
# @File : BERT_w2v2_train.py
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
import torch.optim as optim
from torch.optim import AdamW


from utils_5_wavEnc_textTok import Get_data    # 拿取用于5折的wav数据
from torch.autograd import Variable
from models import BERT_w2v2_GRU
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from transformers import Wav2Vec2Model


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.enabled = False

with open('Train_data_w2v2Large_BERTbaseTokenized_librosaMel.pickle', 'rb') as file:
    data = pickle.load(file)

parser = argparse.ArgumentParser(description="BERT_W2v2GRU_Model")
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--bid_flag', action='store_false')
parser.add_argument('--batch_first', action='store_false')
parser.add_argument('--batch_size', type=int, default=6, metavar='N')   # default 6
parser.add_argument('--log_interval', type=int, default=10, metavar='N')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--optim', type=str, default='AdamW')
parser.add_argument('--attention', action='store_true', default=True)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dia_layers', type=int, default=2)
parser.add_argument('--hidden_layer', type=int, default=256)
parser.add_argument('--out_class', type=int, default=4)
parser.add_argument('--utt_insize', type=int, default=1024)    # base-768   large-1024
args = parser.parse_args()

torch.manual_seed(args.seed)


def Train(epoch):
    train_loss = 0
    model.train()
    total_samples = 0
    correct_samples = 0
    for batch_idx, (data_1, target, ids, att) in enumerate(train_loader):
        if args.cuda:
            data_1, target, ids, att = data_1.cuda(), target.cuda(), ids.cuda(), att.cuda()
        data_1, target = Variable(data_1), Variable(target)
        ids, att = Variable(ids), Variable(att)

        target = target.squeeze()
        utt_optim.zero_grad()
        data_1 = data_1.squeeze()   # torch.Size([6, 48000])
        utt_out = model(data_1, ids, att)
        loss = torch.nn.CrossEntropyLoss()(utt_out, target.long())

        loss.backward()

        utt_optim.step()
        train_loss += loss
        _, predicted = torch.max(utt_out, 1)
        total_samples += target.size(0)
        correct_samples += (predicted == target).sum().item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Accuracy: {:.2f}%'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), train_loss.item() / args.log_interval,
                       (correct_samples / total_samples) * 100
            ))
            train_loss = 0

def Test():
    model.eval()
    label_pre = []
    label_true = []
    with torch.no_grad():
        for batch_idx, (data_1, target,ids, att) in enumerate(test_loader):
            if args.cuda:
                data_1, target, ids, att = data_1.cuda(), target.cuda(), ids.cuda(), att.cuda()
            data_1, target = Variable(data_1), Variable(target)
            ids, att = Variable(ids), Variable(att)
            target = target.squeeze()
            data_1 = data_1.squeeze()
            utt_out = model(data_1, ids, att)
            output = torch.argmax(utt_out, dim=1)
            label_true.extend(target.cpu().data.numpy())
            label_pre.extend(output.cpu().data.numpy())
        accuracy_recall = recall_score(label_true, label_pre, average='macro')
        accuracy_f1 = metrics.f1_score(label_true, label_pre, average='macro')
        CM_test = confusion_matrix(label_true, label_pre)
        print(accuracy_recall)
        print(accuracy_f1)
        print(CM_test)
    return accuracy_f1, accuracy_recall, label_pre, label_true,CM_test

data_save = "BERT_w2v2GRU.txt"

Final_result = []
Fineal_f1 = []
result_label = []
sum_cm = np.zeros((4, 4))
kf = KFold(n_splits=5)
for index, (train, test) in enumerate(kf.split(data)):
    print(f"开启第{index+1}折的训练：")
    train_loader, test_loader, input_test_data_id, input_test_label_org = Get_data(data, train, test, args)
    model = BERT_w2v2_GRU(args)
    #utt_net = Utterance_net(args.utt_insize, args.hidden_layer, args.out_class, args)
    if args.cuda:
        model = model.cuda()

    lr = args.lr
    utt_optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)
    utt_optim = optim.Adam(model.parameters(), lr=lr)
    f1 = 0
    recall = 0
    recall_list = []
    f1_list = []
    cm_list = []
    predict = copy.deepcopy(input_test_label_org)
    for epoch in range(1, args.epochs + 1):
        Train(epoch)
        accuracy_f1, accuracy_recall, pre_label, true_label,cm = Test()
        recall_list.append(accuracy_recall)
        f1_list.append(accuracy_f1)
        cm_list.append(cm)
        if (accuracy_recall > recall):
            num = 0
            for x in range(len(predict)):
                predict[x] = pre_label[num]
                num = num + 1
            result_label = predict
            recall = accuracy_recall
        print("Best Result Until Now:")
        print(recall)

    max_recall = max(recall_list)
    max_f1 = f1_list[recall_list.index(max_recall)]  # 通过在recall列表里检索下标来输出对应的f1数值
    cm = cm_list[recall_list.index(max_recall)]
    sum_cm += cm
    print("成功统计一个混淆矩阵")
    with open(data_save, 'a') as f:
        f.write("第" + str(index + 1) + "折数据：" + "\n" + str(max_recall) + '\n' + str(max_f1) + '\n' + str(cm) + '\n')
        print("输出结果已保存")

    onegroup_result = []
    for i in range(len(input_test_data_id)):
        a = {}
        a['id'] = input_test_data_id[i]
        a['Predictjiu_label'] = result_label[i]
        a['True_label'] = input_test_label_org[i]
        onegroup_result.append(a)
    Final_result.append(onegroup_result)
    Fineal_f1.append(recall)
file = open('Final_result.pickle', 'wb')
pickle.dump(Final_result,file)
file.close()
file = open('Final_f1.pickle', 'wb')
pickle.dump(Fineal_f1,file)
file.close()

with open(data_save, 'a') as f:
    f.write('\n10个最佳混淆矩阵之和是：\n' + str(sum_cm))
    print("最终混淆矩阵：\n",sum_cm)
    print("最终混淆矩阵结果已保存")