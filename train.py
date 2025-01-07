from __future__ import print_function, division
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.utils.data

import torchvision.models as models
from torch.autograd import Variable as V
from torch.nn.parameter import Parameter
import scipy.sparse as sp
import json
from PIL import Image
import argparse
import torchvision.transforms as transforms
import torch.optim as optim
from utils import progress_bar

import clip
import timm
from test import TimmModel

class MultiTaskModel(nn.Module):
    def __init__(self, num_aux1_classes, num_aux2_classes):
        super(MultiTaskModel, self).__init__()
        model = timm.create_model('mobilenetv2_100', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model.children())[:-1])

        # 辅助任务1头
        self.auxiliary_task1_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Linear(512, num_aux1_classes)
        )

        # 辅助任务2头
        self.auxiliary_task2_head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Linear(512, num_aux2_classes)
        )

    def forward(self, x):
        # 使用特征提取器提取特征
        features = self.feature_extractor(x)
        # 将特征展平
        # features = features.view(features.size(0), -1)
        # main_task_output = self.main_task_head(features)
        aux1_task_output = self.auxiliary_task1_head(features)
        aux2_task_output = self.auxiliary_task2_head(features)
        enc_layers = list(self.auxiliary_task1_head.children())
        enc_1 = nn.Sequential(*enc_layers[:1])  # input -> 512
        enc_layers = list(self.auxiliary_task2_head.children())
        enc_2 = nn.Sequential(*enc_layers[:1])  # input -> 512
        return aux1_task_output, aux2_task_output, enc_1(features), enc_2(features)

def Scene_model(img):
    # th architecture to use
    arch = 'resnet18'

    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    model.to(device)
    logit = model(img)
    # print(logit.shape)  # torch.Size([b, 365])
    target_shape = (logit.shape[0], 512)
    padding_size = target_shape[1] - logit.shape[1]
    padding_tensor = torch.zeros(logit.shape[0], padding_size)
    padding_tensor = padding_tensor.to(device)
    result = torch.cat((logit, padding_tensor), dim=1)
    result = result.to(device)
    return result

#######################   GCN

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    rowsum[rowsum==0]=0.0000001
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.00000001
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid*2)
        self.dropout = dropout
        self.fc1 = nn.Linear(nhid *2* 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, x, adj):
        # print('1, ', x.shape)       #torch.Size([batchsize, 6, 256])
        x = F.relu(self.gc1(x, adj))
        # print('1, ', x.shape)       #torch.Size([batchsize, 6, 1024])
        x = F.dropout(x, self.dropout, training=self.training)
        # print('2, ', x.shape)       #torch.Size([batchsize, 6, 1024])
        x = F.relu(self.gc2(x, adj))
        # print('3, ', x.shape)       #torch.Size([batchsize, 6, 2048])
        x = x.view(x.shape[0],1,-1)
        # print('4, ', x.shape)       #torch.Size([batchsize, 1, 12288])
        if x.ndim==2:
            x = x.view(1,-1)
        # print('5, ', x.shape)       #torch.Size([batchsize, 1, 12288])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print('6, ', x.shape)       #torch.Size([batchsize, 1, 8])
        x = x.reshape((x.shape[0], -1))
        # print('7, ', x.shape)       #torch.Size([batchsize, 8])
        return x

emotion_dict = {
    'amusement': 0,
    'awe': 1,
    'contentment': 2,
    'excitement': 3,
    'anger': 4,
    'disgust': 5,
    'fear': 6,
    'sadness': 7,
}

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None, train=True):
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.filenames = []
        self.targets = []

        # with open(data_path, 'r') as f:
        #     lines = f.readlines()
        with open(data_path, 'r') as json_file:
            data = json.load(json_file)

        for sub_list in data:
            if len(sub_list) == 3:
                emotion = sub_list[0]
                image_path = sub_list[1]
                json_path = sub_list[2]
            else:
                print("Invalid sub-list format")

            image_dir = '/home/design/fyx/project/Dataset/EmoSet/'
            filename = os.path.join(image_dir, image_path)
            target = emotion_dict[emotion]
            self.filenames.append(filename)
            self.targets.append(target)
            self.filenames = [os.path.join(image_dir, filename) for filename in self.filenames]


    def __len__(self):
        # TODO: return the length of dataset
        return len(self.filenames)

    def __getitem__(self, index):
        # TODO: load and preprocess the data at index idx
        # and return a tuple (data, label)
        filename, target = self.filenames[index], self.targets[index]

        img = Image.open(filename)

        if self.transform is not None:
            img = self.transform(img)

        return img, int(target)

def main():

    #####################  GCN
    model_GCN = GCN(nfeat=512, nhid=1024, dropout=0.5).to(device)

    #---------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Emotion Classification Training')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    args = parser.parse_args()

    # global best_acc  # best test accuracy
    # best_acc = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainset = EmotionDataset(data_path='/home/design/fyx/project/Dataset/EmoSet/train.json', train=True,
                              transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = EmotionDataset(data_path='/home/design/fyx/project/Dataset/EmoSet/test.json', train=False,
                             transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')

    net = model_GCN



    Attribute_model = MultiTaskModel(11,11)
    checkpoint = torch.load('/home/design/fyx/project/GCN/attribute/checkpoint/new_mv2-2.pth')
    Attribute_model.load_state_dict(checkpoint['net'])

    Attribute_model = Attribute_model.to(device)

    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    clip_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    edges_unordered = np.genfromtxt("cora.cites", dtype=np.int32)  # 读入边的信息
    adj = np.zeros((4, 4))
    for [q, p] in edges_unordered:
        adj[q - 1, p - 1] = 1
    adj = torch.from_numpy(adj)
    adj = normalize(adj)
    adj = torch.from_numpy(adj)
    adj = adj.clone().float()
    adj = adj.to(device)


    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        Attribute_model.eval()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            Scene_f = Scene_model(inputs)
            clip_f = clip_model.encode_image(inputs)
            Brightness_x, Colorfulness_x, Brightness_f, Colorfulness_f = Attribute_model(inputs)

            temp = torch.zeros(Brightness_f.shape[0], 4, Brightness_f.shape[1])
            for num in range(Brightness_f.shape[0]):
                temp[num::] = torch.stack((Scene_f[num, :], Brightness_f[num, :],
                                           Colorfulness_f[num, :], clip_f[num, :]), 0)

            temp = temp.to(device)
            # print(temp.shape)       #torch.Size([16, 6, 256])
            outputs = net(temp, adj)
            # print(outputs.shape)        #torch.Size([16, 8])
            #---------------------------------------------------------------------
            loss = criterion(outputs, targets)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(epoch):
        net.eval()
        global best_acc
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)

                Scene_f = Scene_model(inputs)
                # clip_f = clip_model.encode_image(inputs)
                clip_f = clip_model.encode_image(inputs)
                Brightness_x, Colorfulness_x, Brightness_f, Colorfulness_f = Attribute_model(inputs)

                temp = torch.zeros(Brightness_f.shape[0], 4, Brightness_f.shape[1])
                for num in range(Brightness_f.shape[0]):
                    temp[num::] = torch.stack((Scene_f[num, :], Brightness_f[num, :],
                                               Colorfulness_f[num, :], clip_f[num, :]), 0)

                temp = temp.to(device)
                outputs = net(temp, adj)

                # ---------------------------------------------------------------------
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/test.pt')
            best_acc = acc

    # global best_acc
    for epoch in range(start_epoch, start_epoch + 100):
        train(epoch)
        test(epoch)
        scheduler.step()
        print("best_acc: ", best_acc)

best_acc = 0

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda")
    main()

