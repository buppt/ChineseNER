# coding=utf-8
#encoding: utf-8  
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
print 'Creating the data generator ...'
data_train = BatchGenerator(x_train, y_train, shuffle=True)
data_valid = BatchGenerator(x_valid, y_valid, shuffle=False)
data_test = BatchGenerator(x_test, y_test, shuffle=False)
print 'Finished creating the data generator.'




# Bidirectional LSTM + CRF model.
learning_rate = 0.005
training_epochs = 10
input_size = 1
batch_size = 16
embedding_size = 100
display_num = 5  # 每个 epoch 显示几个结果
batch_num = int(data_train.y.shape[0] / batch_size)  # 每个 epoch 中包含的 batch 数
batch_num_test = int(data_test.y.shape[0] / batch_size)  # 每个 epoch 中包含的 batch 数
display_batch = int(batch_num / display_num)  # 每训练 display_batch 之后输出一次
num_units = len(x_train[0]) # the number of units in the LSTM cell
num_of_word = len(word2id)+1
number_of_classes = len(tag2id)

input_data = tf.placeholder(tf.int32, shape=[batch_size,num_units], name="input_data") # shape = (batch,sentence_len ,input_size)
labels = tf.placeholder(tf.int32,shape=[batch_size,num_units], name="labels") # shape = (batch, sentence)

word_embeddings = tf.get_variable("word_embeddings",[num_of_word, embedding_size])
input_embedded = tf.nn.embedding_lookup(word_embeddings, input_data)
input_embedded = tf.nn.dropout(input_embedded,0.5)

lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1.0, state_is_tuple=True)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(embedding_size, forget_bias=1.0, state_is_tuple=True)
(output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, 
                                                                 lstm_bw_cell, 
                                                                 input_embedded,
                                                                 dtype=tf.float32,
                                                                 time_major=False,
                                                                 scope=None)

bilstm_out = tf.concat([output_fw, output_bw], axis=2)


# Fully connected layer.
W = tf.get_variable(name="W", shape=[batch_size,2 * embedding_size, number_of_classes],
                dtype=tf.float32)

b = tf.get_variable(name="b", shape=[batch_size, num_units, number_of_classes], dtype=tf.float32,
                initializer=tf.zeros_initializer())

bilstm_out = tf.tanh(tf.matmul(bilstm_out, W) + b)

# Linear-CRF.
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(bilstm_out, labels, tf.tile(np.array([num_units]),np.array([batch_size])))

loss = tf.reduce_mean(-log_likelihood)

# Compute the viterbi sequence and score (used for prediction and test time).
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(bilstm_out, transition_params,tf.tile(np.array([num_units]),np.array([batch_size])))

# Training ops.
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss)

saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        for batch in range(batch_num): 
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {input_data:x_batch, labels:y_batch}
            pre,_ = sess.run([viterbi_sequence,train_op], feed_dict)
            acc = 0
            if batch%100==0:
                for i in range(len(y_batch)):
                    for j in range(len(y_batch[0])):
                        if y_batch[i][j]==pre[i][j]:
                            acc+=1
                print float(acc)/(len(y_batch)*len(y_batch[0]))
        path_name = "./model/model"+str(epoch)+".ckpt"
        print path_name
        saver.save(sess, path_name)
        print "model has been saved"
        entityres=[]
        entityall=[]
        for batch in range(batch_num): 
            x_batch, y_batch = data_train.next_batch(batch_size)
            feed_dict = {input_data:x_batch, labels:y_batch}
            pre = sess.run([viterbi_sequence], feed_dict)
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
            feed_dict = {input_data:x_batch, labels:y_batch}
            pre = sess.run([viterbi_sequence], feed_dict)
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


'''                  
with tf.Session() as sess:
    saver.restore(sess, 'data/model9.ckpt')   
    entityres=[]
    entityall=[]
    for batch in xrange(batch_num): 
        x_batch, y_batch = data_train.next_batch(batch_size)
        feed_dict = {input_data:x_batch, labels:y_batch}
        pre = sess.run([viterbi_sequence], feed_dict)
        pre = pre[0]
        entityres = calculate3(x_batch,pre,id2word,id2tag,entityres)
        entityall = calculate3(x_batch,y_batch,id2word,id2tag,entityall)
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
    for batch in xrange(batch_num_test): 
        x_batch, y_batch = data_test.next_batch(batch_size)
        feed_dict = {input_data:x_batch, labels:y_batch}
        pre = sess.run([viterbi_sequence], feed_dict)
        pre = pre[0]
        entityres = calculate3(x_batch,pre,id2word,id2tag,entityres)
        entityall = calculate3(x_batch,y_batch,id2word,id2tag,entityall)
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

'''
