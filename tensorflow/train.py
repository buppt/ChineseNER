# -*- coding: utf-8 -*
import pickle
import pdb

import random as rand
from random import random
import numpy as np
import math
import tensorflow as tf
from Batch import BatchGenerator
import codecs
from resultCal import calculate
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




training_epochs = 10
batch_size = 16

config = {}
config["lr"] = 0.001
config["embedding_dim"] = 100
config["dropout_keep"] = 0.5

config["sen_len"] = len(x_train[0])
config["batch_size"] = batch_size
config["embedding_size"] = len(word2id)+1
config["tag_size"] = len(tag2id)

batch_num = int(data_train.y.shape[0] / batch_size)  
batch_num_test = int(data_test.y.shape[0] / batch_size) 

  
    
def train(model,sess):    
    saver = tf.train.Saver()
    for epoch in range(training_epochs):
        for batch in range(batch_num): 
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {model.input_data:x_batch, model.labels:y_batch}
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
        if True:#epoch%3==0:
            saver.save(sess, path_name)
            print "model has been saved"
            entityres=[]
            entityall=[]
            for batch in range(batch_num): 
                x_batch, y_batch = data_train.next_batch(batch_size)
                feed_dict = {model.input_data:x_batch, model.labels:y_batch}
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
                feed_dict = {model.input_data:x_batch, model.labels:y_batch}
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


model = Model(config)
#saver.restore(sess, 'data/model9.ckpt')  
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train(model,sess)

     

