## set reading type
def read_file(file_path):
    fileobj = open(file_path, 'r', encoding='utf-8')
    samples = []
    tokens = []
    tags = []

    for content in fileobj:
        content = content.strip('\n')
        if content == '-DOCSTART- -X- -X- O':
            pass
        elif content == 'END':
            if len(tokens) != 0:
                samples.append((tokens, tags))
                tokens = []
                tags = []
        else:
            contents = content.split(' ')
            tokens.append(contents[0])
            tags.append(contents[-1])
    return samples
data_samples = read_file('data.txt')
print(data_samples)#我也不知道为什么是空的

#将文本进行分词，并将label转化为数字
## divide label and text
# transform label to num
def get_dicts(datas):
    w_all_dict, n_all_dict = {}, {}
    for sample in datas:
        for token, tag in zip(*sample):
            if token not in w_all_dict.keys():
                w_all_dict[token] = 1
            else:
                w_all_dict[token] += 1

            if tag not in n_all_dict.keys():
                n_all_dict[tag] = 1
            else:
                n_all_dict[tag] += 1

    sort_w_list = sorted(w_all_dict.items(), key=lambda d: d[1], reverse=True)
    sort_n_list = sorted(n_all_dict.items(), key=lambda d: d[1], reverse=True)
    w_keys = [x for x, _ in sort_w_list[:15999]]
    w_keys.insert(0, "UNK",)
    w_keys.insert(1,'PAD')

    n_keys = [x for x, _ in sort_n_list]
    w_dict = {x: i for i, x in enumerate(w_keys)}
    n_dict = {x: i for i, x in enumerate(n_keys)}
    return (w_dict, n_dict)

word_dict, tag_dict = get_dicts(data_samples)
print(word_dict)
print(tag_dict)

#将text(train) 转化为数字，并计算每组的单词个数 n_dict
def w2num(datas, w_dict, n_dict):
    ret_datas = []
    for sample in datas:
        num_w_list, num_n_list = [], []
        for token, tag in zip(*sample):
            if token not in w_dict.keys():
                token = "UNK"

            if tag not in n_dict:
                tag = "O"

            num_w_list.append(w_dict[token])
            num_n_list.append(n_dict[tag])

        ret_datas.append((num_w_list, num_n_list, len(num_n_list)))
    return (ret_datas)
num_data_samples = w2num(data_samples, word_dict, tag_dict)
# print(num_data_samples)

def len_norm(data_num,lens=80):
    ret_datas = []
    for sample1 in list(data_num):
        sample = list(sample1)
        ls = sample[-1]
        #print(sample)
        while(ls<lens):
            sample[0].append(0)
            ls = len(sample[0])
            sample[1].append(0)
        else:
            sample[0] = sample[0][:lens]
            sample[1] = sample[1][:lens]

        ret_datas.append(sample[:2])
    return(ret_datas)


normalized_data = len_norm(num_data_samples, lens=80)
print(normalized_data)
#这部分到时候补上LUTAO的代码，就是要把这一部分写的漂亮一些

from tqdm import tqdm
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import *
from keras.optimizers  import *
from keras.utils import np_utils
import numpy as np
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
#这个放在公司电脑上面：pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

data_path="./conll2003_v2/"
data_parts = ['train', 'valid', 'test']
extension = '.txt'
dataset = {}
for data_part in tqdm(data_parts):
    file_path = data_path + data_part + extension
    dataset[data_part] = read_file(str(file_path))
train=dataset['train']
test=dataset['test']
valid=dataset['valid']

## transform train into w_dict, n_dict
w_dict,n_dict = get_dicts(dataset['train'])

# 将text(train) 转化为数字，并计算每组的单词个数 n_dict
data_num = {}
data_num["train"] = w2num(train,w_dict,n_dict)

## 将text , label 补成等长的 array , 长度为80
data_norm = {}
data_norm["train"] = len_norm(data_num["train"])
from keras.layers import Embedding ,Bidirectional,LSTM,GRU,TimeDistributed,Dense, BatchNormalization
# from keras_contrib.layers import CRF
from tensorflow_addons.layers import CRF
num_classes=len(n_dict.keys())

model = Sequential()
model.add(Embedding(16000, 128, input_length=80))
model.add(Bidirectional(GRU(64,return_sequences=True),merge_mode="concat"))
model.add(Bidirectional(GRU(64,return_sequences=True),merge_mode="concat"))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
crf = CRF(len(n_dict.keys()), sparse_target=True)
model.add(crf)

print(model.summary())

opt = Adam(5e-4)
#model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
model.compile(loss=crf.loss_function,optimizer=opt,metrics=[crf.accuracy])

# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(loss=crf.loss_function, optimizer=sgd, metrics=[crf.accuracy])
## set class_weights
import numpy as np
import pandas as pd
train_data = np.array(data_norm["train"])
train_y = train_data[:,1,:]
np.unique(train_y)
data_list = map(lambda x: x[1], train_y)
train_ser = pd.Series(data_list)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',np.unique(train_y),train_ser)
# 这一段很重要
# 正式训练模型
train_x = train_data[:,0,:]  #第一列为x
train_y = train_data[:,1,:]  #第二列为y
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))  #reshape重组array 3_dim
print(train_y.shape)


## valid set
va_w_dict,va_n_dict = get_dicts(dataset['valid'])
data_num["valid"] = w2num(valid,va_w_dict,va_n_dict)
data_norm["valid"] = len_norm(data_num["valid"])
valid_data = np.array(data_norm["valid"])
valid_x = valid_data[:,0,:]  #第一列为x
valid_y = valid_data[:,1,:]  #第二列为y
valid_y = valid_y.reshape((valid_y.shape[0], valid_y.shape[1], 1))
print(valid_y.shape)

model.fit(x=train_x,y=train_y,epochs=10,batch_size=32,
          class_weight = class_weights,
          verbose=1,
          validation_data=(valid_x, valid_y),
          shuffle=True)
# save the weigths
# model.load_weights("model.h5")
x=568
pre_y = model.predict(train_x[x:x+1])
# print(pre_y.shape)

# model predict
pre_y = np.argmax(pre_y,axis=-1)
print(pre_y)
print(train_y[x])
# for i in range(0,len(train_y[3:4])):
#     print("label "+str(i),train_y[i])









