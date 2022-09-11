# -*- coding: utf-8 -*-

import copy

import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_processor
import logging
import torchmetrics


fileName = './model_train.log'
handler = [logging.FileHandler(filename=fileName,encoding="utf-8")]
logging.basicConfig(level = logging.DEBUG, handlers = handler)
torch.manual_seed(123) #保证每次运行初始化的随机数相同

vocab_size = 5000   #词表大小
embedding_size = 64   #词向量维度
num_classes = 2    #6分类 todo
sentence_max_len = 64  #单个句子的长度
hidden_size = 16

num_layers = 1  #一层lstm
num_directions = 2  #双向lstm
lr = 1e-3
batch_size = 16   # batch_size 批尺寸
epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app_name = "测试"
bug_type = ["隐私不相关", "隐私相关"]
# lexicon = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[]}
lexicon = {0: [], 1: []}
n = 500 # 选择置信度最高的前n条数据
m = 500 # 选择注意力权重最高的前m个词

t1 = 3
t2 = 8
threshold_confidence = 0.9

#Bi-LSTM模型
class BiLSTMModel(nn.Module):
    # 声明带有模型参数的层
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        # torch.nn.Sequential 类是 torch.nn 中的一种序列容器，通过在容器中嵌套各种实现神经网络中具体功能相关的类，来完成对神经网络模型的搭建，
        # 最主要的是，参数会按照我们定义好的序列自动传递下去。
        # torch.nn.Linear 类接收的参数有三个，分别是输入特征数、输出特征数和是否使用偏置，
        # 设置是否使用偏置的参数是一个布尔值，默认为 True ，即使用偏置。
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # 从hidden_size到hideen_size的线性变换
            nn.ReLU(inplace=True) # 激活函数
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        # print()
        self.act_func = nn.Softmax(dim=1)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        #lstm的输入维度为 [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size] ，将x进行依次转置
        
        #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)
        
        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        #out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_laye，rs * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        
        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        # Bi-LSTM + Attention 就是在Bi-LSTM的模型上加入Attention层，在Bi-LSTM中我们会用最后一个时序的输出向量 作为特征向量，然后进行softmax分类。Attention是先计算每个时序的权重，然后将所有时序 的向量进行加权和作为特征向量，然后进行softmax分类。在实验中，加上Attention确实对结果有所提升。
        # https://blog.csdn.net/zwqjoy/article/details/96724702
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]   [16, 1, 16]
        # print(attention_w)
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len] [16 ,1, 32]
        # print(attention_context)
        
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len],权重归一化 [16, 1, 32]
        # print(softmax_w) # 这个是注意力机制的权重向量
        
        
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.liner(x)
        # print("self.liner(x).shape", x.shape)
        x = self.act_func(x) # [16, 2]
        # print("softmax_w.shape", softmax_w.shape)
        # print(x.shape)
        # print("X", x)
        return softmax_w, x

# 将 发展集中新预测的标签数据添加到训练集中，然后再次训练分类器
# 这些新伪标签数据的类别分布要平衡

# 通过基础分类器和词库共同 预测 标签的类型。
# 这种预测 共分为两个流程：第一个是 预测发展集的标签并把预测好的数据加到训练集中
# 第二个是 当加完所有的伪标签数据后，重新训练 基础分类器，用 新的基础分类器+最全的词库去预测 测试集
def develop_to_train(new_labeled_data, train_features, develop_features, train_labels, develop_labels):
    for key in sorted(new_labeled_data, reverse = True):
        feature = develop_features.pop(key)
        
        del develop_labels[key]
        label_index = new_labeled_data[key]
        label = [0] * num_classes
        label[label_index] = 1
        train_labels.append(label)
        train_features.append(feature)               
    return train_features, develop_features, train_labels, develop_labels

# 在发展集上重新运行基础分类器，获得一组关于发展集的关键词
# 方法 是 通过基础分类器在发展集上预测出的置信度和单词的attention 为发展集收集词库
def test_with_lexicon(model, develop_loader, develop_feature_origin, word2index):
    model.eval() # 评估模式而非训练模式,batchNorm层和dropout层等用于优化训练而添加的网络层会被关闭，使评估时不会发生偏移

    confidence_list = [] #总的置信度列表
    category_list = [] # 总的预判种类列表
    attention_list = [] # 总的word权重列表

    for datas, labels in develop_loader:
        datas = datas.to(device) # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
        softmax_w, preds = model.forward(datas)
        softmax_w = softmax_w.squeeze(dim = 1) # [6, 64]
        attention = softmax_w.tolist()
        attention_list.extend(attention)
        print("label_test_with_lexicon:", labels)
        print("preds_test_with_lexicon:", preds)
        
        a = preds.max(dim = 1)
        confidence = a[0].tolist()  # 置信度列表
        category = a[1].tolist()  # 预测的类别列表

        confidence_list.extend(confidence)
        category_list.extend(category)


    confidence_dict = dict(zip(confidence_list, list(range(len(confidence_list)))))
    category_dict = dict(zip(list(range(len(category_list))),category_list))
    attention_dict = dict(zip(list(range(len(attention_list))),attention_list))
    # print("confidence_dict", confidence_dict)
    print("category_dict:", category_dict)

    lexicon_num = {0: 0, 1: 0}
    # print("sorted(confidence_dict, reverse = True)", sorted(confidence_dict, reverse = True))
    for i in sorted(confidence_dict, reverse = True):
        # print("i:",i)
        # print("confidence_dict[i]:", confidence_dict[i])
        # print("category_dict:", category_dict)
        lexicon_key = category_dict[confidence_dict[i]]
        # print("lexicon_key", lexicon_key)
        if lexicon_num[lexicon_key] <= n: # 每个类别取置信度最高的前n条数据
            # print(str(lexicon_key) + ":" + str(i))
            lexicon_num[lexicon_key] += 1
            lexicon_value_attention = attention_dict[confidence_dict[i]]
            lexicon_value_word = develop_feature_origin[confidence_dict[i]]
            attention2word = dict(zip(lexicon_value_attention,lexicon_value_word))
            print("attention2word", attention2word)

            word2attention = {}
            for j in sorted(attention2word, reverse = True):
                word = list(word2index.keys())[list(word2index.values()).index(attention2word[j])]
                # print("word", word)
                if word != "<unk>" and word != "<pad>":
                    if word in word2attention.keys():
                        word2attention[word] += j
                    else:
                        word2attention[word] = j
            q = 0
            print("word2attention", word2attention)
            for k in sorted(word2attention.items(), key = lambda kv:(kv[1], kv[0]), reverse = True):
                if q < m and k[0] not in lexicon[lexicon_key]:
                    lexicon[lexicon_key].append(k[0])
                q += 1
            
    print("lexicon_num:",lexicon_num)
    new_labeled_data = {}
    
    # 此时，已经获得了这一轮的类别词库
    # 记录下新被贴标签的数据，记录第k个数据和它新的类别，之后在发展集中剔除它，把它加到训练集
    for k in range(len(confidence_list)):
        lexicon_value_word = develop_feature_origin[k]
        match_num = [0] * num_classes
        for value_word in lexicon_value_word:
            word = list(word2index.keys())[list(word2index.values()).index(value_word)]
            if word != "<unk>" and word != "<pad>":
                for l in range(num_classes):
                    if word in lexicon.get(l):
                        #会不会出现同一个词在多个类别词库中出现的问题
                        match_num[l] = match_num[l] + 1 
        max_num = max(match_num)
        # print(str(match_num) + "---"+ str(confidence_list[k]))
        if match_num.count(max_num) != 1:
            continue
        elif max_num >= t2:
            # 就根据词库对应的类贴标签给这个数据
            new_labeled_data[k] = match_num.index(max_num)
        elif confidence_list[k] > threshold_confidence and max_num >= t1 and max_num < t2:
            new_labeled_data[k] = category_list[k]

    # 返回被标记的数据的行数和它的新类别
    return new_labeled_data
        
def test(model, test_loader, loss_func, test_feature_origin, word2index):
    model.eval()
    loss_val = 0.0
    corrects = 0.0

    confidence_list = [] #总的置信度列表
    category_list = [] # 总的预判种类列表
    label_list = []
    
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)

        labels_num = labels.tolist()
        label_list_tmp = []
        for label in labels_num:
            sum_label = 0
            for i in range(len(label)):
                sum_label = sum_label + label[i] * i
            label_list_tmp.append(sum_label)
        
        softmax_w, preds = model.forward(datas)

        a = preds.max(dim = 1)
        confidence = a[0].tolist() # 置信度列表
        category = a[1].tolist() # 预测的类别列表

        confidence_list.extend(confidence)
        category_list.extend(category)
        label_list.extend(label_list_tmp)

        """
        loss = loss_func(preds, labels)
        loss_val += loss.item() * datas.size(0)
        
        #获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)

        labels = torch.argmax(labels, dim=1)
        corrects += torch.sum(preds == labels).item()
        """
    for k in range(len(confidence_list)):
        lexicon_value_word = test_feature_origin[k]
        match_num = [0] * num_classes
        for value_word in lexicon_value_word:
            word = list(word2index.keys())[list(word2index.values()).index(value_word)]
            if word != "<unk>" and word != "<pad>":
                for l in range(num_classes):
                    if word in lexicon.get(l):
                        #会不会出现同一个词在多个类别词库中出现的问题
                        match_num[l] = match_num[l] + 1
        max_num = max(match_num)
        if match_num.count(max_num) != 1:
            continue
        elif max_num >= t2:
            # 就根据词库对应的类贴标签给这个数据
            category_list[k] = match_num.index(max_num)

    test_loss = 0
    test_acc = 1

    for i in range(len(category_list)):
        # print("第{}个标签: category_list: {}, label_list: {}".format(i, category_list[i], label_list[i]))
        if category_list[i] == label_list[i]:
            corrects = corrects + 1
    test_acc = corrects / len(category_list)        
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc

def test_origin(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    test_pre = torchmetrics.Precision(num_classes=2)
    test_recall = torchmetrics.Recall(num_classes=2)
    test_f1 = torchmetrics.F1Score(num_classes=2)
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        labels_recall = labels
        # print("datas", datas)
        print("label_test_origin", labels)
        softmax_w, preds = model.forward(datas)
        print("测试预测结果preds", preds)
        # print("preds_test_origin", preds)
        loss = loss_func(preds, labels)
        loss_val += loss.item() * datas.size(0)
        #获取预测的最大概率出现的位置

        preds = torch.argmax(preds, dim=1)
        labels = torch.argmax(labels, dim=1)

        corrects += torch.sum(preds == labels).item()
        test_pre(preds.cpu(), labels.cpu())
        test_recall(preds.cpu(), labels.cpu())
        test_f1(preds.cpu(), labels.cpu())

    print("LABELS:", labels)
    print("PREDS:", preds)
    # print(test_pre(preds.cpu(), labels.cpu()))
    total_pre = test_pre.compute()
    total_recall = test_recall.compute()
    total_f1 = test_f1.compute()

    print("PRECISION:", total_pre)
    print("RECALL", total_recall)
    print("F1", total_f1)

    racall = sklearn.metrics.recall_score(labels.cpu(), preds.cpu(), average="macro", zero_division = 0)
    print("召回率为:", racall)
    # logging.info("召回率为{}".format(racall))
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc

def train_origin(model, train_loader, test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            # print(labels.shape)
                        
            attention_w, preds = model.forward(datas) # 使用model预测数据
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            print("训练中的预测preds", preds)
            print("训练中的lables", labels)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)

        if(epoch % 2 == 0):
            # print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            # print("训练中的测试结果")
            test_acc = test_origin(model, train_loader, loss_func)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

# 从给定的训练集数据中创建一个基础分类器，训练集数据很少，数据类别分布平衡
# 这个基础分类器过度拟合训练集？？？？？？？ todo
def train(model, train_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())

    # epoch、batch、iteration的概念 https://www.jianshu.com/p/22c50ded4cf7?from=groupmessage
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:

            datas = datas.to(device)
            labels = labels.to(device)
            
            # print("第{}批训练数据: labels: {}".format(epoch, labels))
            
            attention_w, preds = model.forward(datas) # 使用model预测数据
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            labels = torch.argmax(labels, dim=1)
            corrects += torch.sum(preds == labels).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)

        #print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
        if best_val_acc < train_acc:
            best_val_acc = train_acc
            best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

if __name__ == "__main__":    
    processor = data_processor.DataProcessor(dataset_name = app_name)
    train_features, develop_features, test_features, train_labels, develop_labels, test_labels, word2index = processor.get_datasets_origin(vocab_size=vocab_size, max_len=sentence_max_len)
    print("train_features", train_features)
    # print("train_labels", train_labels)
    train_datasets, develop_datasets, test_datasets = processor.get_datasets(train_features, develop_features, test_features, train_labels, develop_labels, test_labels, vocab_size=vocab_size, embedding_size=embedding_size)
    # for datas, labels in train_datasets:
    #     print("datas", datas)
    #     print("labels", labels)
    # print("train_datasets", train_datasets)

    # train_loader是 batch_size(16)个 数据(train_features)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
    develop_loader = torch.utils.data.DataLoader(develop_datasets, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False)
    
    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.BCELoss()
    # 训练基础的模型
    logging.info("开始训练基础分类器")
    # model = train(model, train_loader, optimizer, loss_func, epochs)
    # print("开始训练基础分类器")
    model = train_origin(model, train_loader, test_loader, optimizer, loss_func, epochs)
    # print("11111111")
    test_acc = test_origin(model, test_loader, loss_func)
    logging.info("初始分类器准确率为{}".format(test_acc))
    
    # i = 0
    """while 1:
        # i = i+ 1
        # print("iiiiii:", i)
        # 从发展集中构建词库
        new_labeled_data = test_with_lexicon(model, train_loader, train_features, word2index)

        # print("重新贴标签的数据是{}".format(new_labeled_data))
        print("现在的词库是{}".format(lexicon))
    
        if len(new_labeled_data) == 0:
            print("new_labeled_data", new_labeled_data)
            break"""
    new_labeled_data = test_with_lexicon(model, test_loader, test_features, word2index)
        # train_features, develop_features, train_labels, develop_labels = develop_to_train(new_labeled_data, train_features, develop_features, train_labels, develop_labels)
        # print("AAAAA")
        # embed = nn.Embedding(vocab_size + 2, embedding_size) # https://www.jianshu.com/p/63e7acc5e890
        #
        # train_features_after1 = torch.LongTensor(train_features)
        # train_features_after1 = embed(train_features_after1)
        # train_features_after2 = Variable(train_features_after1, requires_grad=False)
        # train_labels_after = torch.FloatTensor(train_labels)
        # train_datasets = torch.utils.data.TensorDataset(train_features_after2, train_labels_after)
        # train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=False)
        #
        # develop_features_after1 = torch.LongTensor(develop_features)
        # develop_features_after1 = embed(develop_features_after1)
        # develop_features_after2 = Variable(develop_features_after1, requires_grad=False)
        # develop_labels_after = torch.FloatTensor(develop_labels)
        # develop_datasets = torch.utils.data.TensorDataset(develop_features_after2, develop_labels_after)
        # develop_loader = torch.utils.data.DataLoader(develop_datasets, batch_size=batch_size, shuffle=False)
        #
        # logging.info("开始第{}次重训练".format(i))
        # model = train_origin(model, train_loader, test_loader, optimizer, loss_func, epochs)
     
    print("22222")
    model = train_origin(model, train_loader, test_loader, optimizer, loss_func, epochs)
    test_acc = test_origin(model, test_loader, loss_func)
    logging.info("训练完成，测试集准确率为{}".format(test_acc))
    print("现在的词库是1{}".format(lexicon))
    print("333333")
    lexicon = {0: [], 1: []}
    new_labeled_data = test_with_lexicon(model, test_loader, test_features, word2index)
    print("现在的词库是2{}".format(lexicon))
    logging.info("现在的词库是{}".format(lexicon))
    torch.save(model, "../classify_model/" + app_name + ".pth")
