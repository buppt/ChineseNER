# coding=utf-8
import pickle
import pdb
with open('../data/Bosondata.pkl', 'rb') as inp:
	word2id = pickle.load(inp)
	id2word = pickle.load(inp)
	tag2id = pickle.load(inp)
	id2tag = pickle.load(inp)
	x_train = pickle.load(inp)
	y_train = pickle.load(inp)
	x_test = pickle.load(inp)
	y_test = pickle.load(inp)
	x_valid = pickle.load(inp)
	y_valid = pickle.load(inp)
print "train len:",len(x_train)
print "test len:",len(x_test)
print "valid len", len(x_valid)

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import codecs 
from BiLSTM_CRF import BiLSTM_CRF

#############
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 100
HIDDEN_DIM = 200
EPOCHS = 10

tag2id[START_TAG]=len(tag2id)
tag2id[STOP_TAG]=len(tag2id)


model = BiLSTM_CRF(len(word2id)+1, tag2id, EMBEDDING_DIM, HIDDEN_DIM)

optimizer = optim.SGD(model.parameters(), lr=0.005, weight_decay=1e-4)


def calculate(x_test,y_test,epoch):
    entityres=[]
    entityres_pre=[]
    i=0
    for x,y in zip(x_test,y_test):
        i+=1
        if i%300==0:
            print i
        x=torch.tensor(x, dtype=torch.long)
        score,predict = model(x)

        j=0
        while(j<len(y)):
            if x[j]==0:
                j+=1
                continue
            else:
                if j<len(y) and (y[j]>len(id2tag) or y[j]==0):
                    j+=1
                    continue
               
                if j<len(y) and id2tag[y[j]][0]=='B':
                    entitytype=id2tag[y[j]][2:]
                    entity=[id2word[x[j]]]
                    j+=1
                    
                    while(j<len(y) and y[j]!=0 and id2tag[y[j]][0]=='M' and id2tag[y[j]][2:]==entitytype):
                        entity.append(id2word[x[j]])
                        j+=1
                        
                        
                    if j<len(y) and y[j]!=0 and id2tag[y[j]][0]=='E' and id2tag[y[j]][2:]==entitytype:
                        entity.append(id2word[x[j]])
                        entity.append(entitytype)
                        entity.append(j)
                        entityres.append(entity)
                        j+=1
                        
                j+=1


        j=0
        while(j<len(predict)):
            if x[j]==0:
                j+=1
                continue
            else:
                if j<len(predict) and (predict[j]>len(id2tag) or predict[j]==0):
                    j+=1
                    continue
                if j<len(predict) and id2tag[predict[j]][0]=='B':
                    entitytype=id2tag[predict[j]][2:]
                    entity=[id2word[x[j]]]
                    j+=1
                    
                    while(j<len(predict) and predict[j]!=0 and id2tag[predict[j]][0]=='M' and id2tag[predict[j]][2:]==entitytype):
                        entity.append(id2word[x[j]])
                        j+=1
                        
                    if j<len(predict) and predict[j]!=0 and id2tag[predict[j]][0]=='E' and id2tag[predict[j]][2:]==entitytype:
                        entity.append(id2word[x[j]])
                        entity.append(entitytype)
                        
                        entity.append(j)
                        entityres_pre.append(entity)
                        j+=1
                        
                j+=1                
    jiaoji = [i for i in entityres_pre if i in entityres]
    if len(jiaoji)!=0:
        zhun = float(len(jiaoji))/len(entityres_pre)
        zhao = float(len(jiaoji))/len(entityres)
        print "zhun:", zhun
        print "zhao:", zhao
        print "f:", (2*zhun*zhao)/(zhun+zhao)
    else:
        print "zhun:",0


st = ""
for epoch in range(EPOCHS):
    index=0
    for sentence, tags in zip(x_train,y_train):
        index+=1
        model.zero_grad()

        sentence=torch.tensor(sentence, dtype=torch.long)
        tags = torch.tensor([tag2id[t] for t in tags], dtype=torch.long)

        loss = model.neg_log_likelihood(sentence, tags)

        loss.backward()
        optimizer.step()
        if index%300==0:
            print "epoch",epoch,"index",index

    calculate(x_test,y_test,epoch)
    print "epoch:",epoch


torch.save(model, "./model/model.pkl")
print "model has been saved"

