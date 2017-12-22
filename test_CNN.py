# -*- coding: utf-8 -*-
#coding = utf-8
import numpy as np
from CNN_model import MNIST_CNN
import tensorflow as tf
import os
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_class', 10, 'number of class')
tf.app.flags.DEFINE_float("learning_rate",0.001,"learning rate")
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('decay_steps', 256, 'decay leaning rate after how many batch')
tf.app.flags.DEFINE_float("decay_rate",0.9,"decay rate of learning rate")
tf.app.flags.DEFINE_float("l2_lambda",0.0005,"l2_lambda")
tf.app.flags.DEFINE_integer('input_size', 784, 'size of an image input  28*28=784')
tf.app.flags.DEFINE_integer('num_channels', 1, 'number of channels of input')#input is gray image
tf.app.flags.DEFINE_integer('conv1_filter', 32, 'how many filter in conv1')#LeNet has 6
tf.app.flags.DEFINE_integer('kernel1_size', 5, 'number of kernel in conv1')#same with LeNet
tf.app.flags.DEFINE_integer('conv2_filter', 64, 'number of filter in conv2') #LeNet has 16
tf.app.flags.DEFINE_integer('kernel2_size', 5, 'nnumber of kernel in conv2')#same with LeNet
tf.app.flags.DEFINE_integer('fc_size', 512, 'fully connection layer size') 
tf.app.flags.DEFINE_float("dropout_keep_prob",0.8,"dropout_keep_prob")
tf.app.flags.DEFINE_boolean("is_training",True,"is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_string('data_path', os.path.join(os.path.abspath('..'), 'workspace', 'data'), 'location of data')
tf.app.flags.DEFINE_string('checkpoint_path', os.path.join(os.path.abspath('..'), 'workspace', 'checkpoint', 'CNN'), 'location of checkpoint file')
tf.app.flags.DEFINE_string('result_path', os.path.join(os.path.abspath('..'), 'workspace', 'result'), 'location of result file')
tf.app.flags.DEFINE_string('result_name', 'CNN', 'name of result file')
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每1轮做一次验证
tf.app.flags.DEFINE_integer("num_epochs",30,"number epoch")



def test():
    #1 load data
    data_file = open(os.path.join(FLAGS.data_path, 'target.pkl'), 'r')
    trainX, validationX, testX, trainY, validationY, testY = pickle.load(data_file)
    data_file.close()

    #2 create session
    config=tf.ConfigProto()#set parameter of session
    #how many GPU can be used
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #devote few resource at the beginning, and it will increase with the more need from task
    config.gpu_options.allow_growth=True
    with tf.Session(config = config) as sess:
        CNN = MNIST_CNN(FLAGS.num_class,FLAGS.learning_rate,FLAGS.batch_size,FLAGS.decay_steps,FLAGS.decay_rate,FLAGS.l2_lambda, \
            FLAGS.input_size, FLAGS.num_channels, FLAGS.conv1_filter, FLAGS.kernel1_size, FLAGS.conv2_filter, FLAGS.kernel2_size, \
            FLAGS.fc_size, FLAGS.dropout_keep_prob, FLAGS.is_training)
        #creat the checkpoint file or load pretrained model
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(FLAGS.checkpoint_path , "checkpoint")):
            print("Restoring Variables from Checkpoint for cnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_path ))
        else:
            print('cant find the checkpoint')
            return 

        #3 feed data and training 
        num_of_test_data = len(testX)
        batch_size = FLAGS.batch_size

        loss, acc, counter = 0.0, 0.0, 0
        for start, end in zip(range(0, num_of_test_data, batch_size), range(batch_size, num_of_test_data, batch_size)):
            curr_loss, curr_acc = sess.run([CNN.loss, CNN.acc], feed_dict={CNN.input_x:testX[start:end], CNN.input_y:testY[start:end]})
            loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
        print("Test Loss:%.3f\tTest Accuracy:%.3f" %(loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
         


def main(argv=None):
    test()

if __name__ == '__main__':
    tf.app.run()
