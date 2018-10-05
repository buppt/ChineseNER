# -*- coding: utf-8 -*
import pickle
import pdb
import codecs
import re
import sys
import math



import numpy as np

import tensorflow as tf
from Batch import BatchGenerator
from resultCal import calculate,get_entity
from bilstm_crf import Model



with open('../data/renmindata.pkl', 'rb') as inp:
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
print 'Creating the data generator ...'
data_train = BatchGenerator(x_train, y_train, shuffle=True)
data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
data_test = BatchGenerator(x_test, y_test, shuffle=False)
print 'Finished creating the data generator.'

word2vec = {}
with codecs.open('vec.txt','r','utf-8') as input_data:   
    for line in input_data.readlines():
        word2vec[line.split()[0]] = map(eval,line.split()[1:])

embedding_pre = []
unknow_pre = []
unknow_pre.extend([0]*100)
embedding_pre.append(unknow_pre) #wordvec id 0
for word in word2id:
    if word2vec.has_key(word):
        embedding_pre.append(word2vec[word])
    else:
        embedding_pre.append(unknow_pre)

embedding_pre = np.asarray(embedding_pre)


training_epochs = 21
batch_size = 32

config = {}
config["lr"] = 0.001
config["embedding_dim"] = 100


config["sen_len"] = len(x_train[0])
config["batch_size"] = batch_size
config["embedding_size"] = len(word2id)+1
config["tag_size"] = len(tag2id)

batch_num = int(data_train.y.shape[0] / batch_size)  
batch_num_test = int(data_test.y.shape[0] / batch_size) 


    
def train(model,sess):    
    for epoch in range(training_epochs):
        for batch in range(batch_num): 
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {model.input_data:x_batch, model.labels:y_batch, model.embedding_placeholder:embedding_pre}
            pre,_ = sess.run([model.viterbi_sequence,model.train_op], feed_dict)
            acc = 0
            if batch%100==0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j]==pre[i][j]:
                            acc+=1
                print float(acc)/(len(y_batch)*len(y_batch[0]))
        path_name = "./model/model"+str(epoch)+".ckpt"
        print path_name
        if epoch%2==0:
            saver.save(sess, path_name)
            print "model has been saved"
            entityres=[]
            entityall=[]
            for batch in range(batch_num): 
                x_batch, y_batch = data_train.next_batch(batch_size)
                feed_dict = {model.input_data:x_batch, model.labels:y_batch, model.embedding_placeholder:embedding_pre}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch,pre,id2word,id2tag,entityres)
                entityall = calculate(x_batch,y_batch,id2word,id2tag,entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji)!=0:
                zhun = float(len(jiaoji))/len(entityres)
                zhao = float(len(jiaoji))/len(entityall)
                print "train"
                print "zhun:", zhun
                print "zhao:", zhao
                print "f:", (2*zhun*zhao)/(zhun+zhao)
            else:
                print "zhun:",0

            entityres=[]
            entityall=[]
            for batch in range(batch_num_test): 
                x_batch, y_batch = data_test.next_batch(batch_size)
                feed_dict = {model.input_data:x_batch, model.labels:y_batch, model.embedding_placeholder:embedding_pre}
                pre = sess.run([model.viterbi_sequence], feed_dict)
                pre = pre[0]
                entityres = calculate(x_batch,pre,id2word,id2tag,entityres)
                entityall = calculate(x_batch,y_batch,id2word,id2tag,entityall)
            jiaoji = [i for i in entityres if i in entityall]
            if len(jiaoji)!=0:
                zhun = float(len(jiaoji))/len(entityres)
                zhao = float(len(jiaoji))/len(entityall)
                print "test"
                print "zhun:", zhun
                print "zhao:", zhao
                print "f:", (2*zhun*zhao)/(zhun+zhao)
            else:
                print "zhun:",0

                

def test_input(model,sess):
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt is None:
        print 'Model not found, please train your model first'
        return
    path = ckpt.model_checkpoint_path
    print 'loading pre-trained model from %s.....' % path
    saver.restore(sess, path)
    
    max_len = 60
    def padding(ids):
        if len(ids) >= max_len:  
            return ids[:max_len]
        else:
            ids.extend([0]*(max_len-len(ids)))
            return ids
    while True:
        text = raw_input("Enter your input: ").decode('utf-8');
        text = re.split(u'[，。！？、‘’“”（）]', text) 
        text_id=[]
        for sen in text:
            word_id=[]
            for word in sen:
                if word in word2id:
                    word_id.append(word2id[word])
                else:
                    word_id.append(word2id["unknow"])
            text_id.append(padding(word_id))
        zero_padding=[]
        zero_padding.extend([0]*max_len)
        text_id.extend([zero_padding]*(batch_size-len(text_id)))    
        feed_dict = {model.input_data:text_id}
        pre = sess.run([model.viterbi_sequence], feed_dict)
        entity = get_entity(text,pre[0],id2tag)
        print 'result:'
        for i in entity:
            print i
        '''
        while True:
        if len(text_id)>=batch_size:
            model.pre
            text_id = text_id[batch_size-1:]
        else:
            text_id.extend(zero_padding*(batch_size-len(text_id)))
        '''


if len(sys.argv)==2 and sys.argv[1]=="test":
    print "begin to test..."
    model = Model(config,dropout_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        test_input(model,sess)
else:
    print "begin to train..."
    model = Model(config,dropout_keep=0.5)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  
        train(model,sess)
        
     

