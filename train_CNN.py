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
tf.app.flags.DEFINE_integer("validate_every", 1, "validate every validate_every epochs.") #每1轮做一次验证
tf.app.flags.DEFINE_integer("num_epochs",20,"number epoch")



def train():
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    file_write = open(os.path.join(FLAGS.result_path, FLAGS.result_name), 'wb')
    try:
        file_write.write('batch_size: %d\tconv1_filter: %d\tkernel1_size: %d\tconv2_filter: %d\tkernel2_size: %d\tnum_epochs: %d\tfc_size: %d\tl2_lambda: \
        %f\tdropout_keep_prob: %f\tlearning_rate:%f\n' 
            % (FLAGS.batch_size, FLAGS.conv1_filter, FLAGS.kernel1_size, FLAGS.conv2_filter, FLAGS.kernel2_size, FLAGS.num_epochs, FLAGS.fc_size, \
                FLAGS.l2_lambda, FLAGS.dropout_keep_prob, FLAGS.learning_rate))
    finally:
        file_write.close()
    #1 load data
    print 'load data'
    data_file = open(os.path.join(FLAGS.data_path, 'source.pkl'), 'r')
    trainX, validationX, testX, trainY, validationY, testY = pickle.load(data_file)
    trainX = np.array(trainX)
    trainY = np.array(trainY)
    validationX = np.array(validationX)
    validationY = np.array(validationY)
    testX = np.array(testX)
    testY = np.array(testY)
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
            write_result("Restoring Variables from Checkpoint for cnn model.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.checkpoint_path ))
        else:
            print('Initializing Variables')
            write_result('Initializing Variables')
            sess.run(tf.global_variables_initializer())
        curr_epoch=sess.run(CNN.epoch_step) #epoch_step's initial value is 0

        #3 feed data and training 
        num_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        last_acc = 0
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, num_of_training_data, batch_size), range(batch_size, num_of_training_data, batch_size)):
                curr_loss, curr_acc, _ = sess.run([CNN.loss, CNN.acc, CNN.train_op], feed_dict={CNN.input_x:trainX[start:end], \
                    CNN.input_y:trainY[start:end]})
                loss, acc, counter = loss + curr_loss, acc + curr_acc, counter + 1
                if counter % 100 == 0:  #every 10 batch output the loss and acc
                    write_result('Epoch: ' + str(epoch)+ '\tBatch: '+ str(counter)+ '\tTrain Loss: '+ str(loss/float(counter))+ '\tTrain Accuracy: ' + str(acc/float(counter)))
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch,counter,loss/float(counter),acc/float(counter))) #tTrain Accuracy:%.3f---》acc/float(counter)
            write_result("going to increment epoch counter....")
            print("going to increment epoch counter....")
            sess.run(CNN.epoch_increment)  #add 1 to epoch   
            
            #4 validate
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))   
            if epoch % FLAGS.validate_every==0:
                eval_loss, eval_acc=do_eval(sess,CNN,validationX,validationY,batch_size)
                write_result('Epoch: '+ str(epoch)+ '\tvalidate Loss: '+ str(eval_loss)+ '\tvalidate Accuracy: ' + str(eval_acc))
                print("Epoch %d validate Loss:%.3f\tvalidate Accuracy: %.3f" % (epoch,eval_loss,eval_acc))
                #save model to checkpoint
                if eval_acc > last_acc:
                    save_path=os.path.join(FLAGS.checkpoint_path,"model.ckpt")
                    saver.save(sess,save_path,global_step=epoch)
                    last_acc = eval_acc           


# 在validate集上做验证，报告损失、精确度
def do_eval(sess,CNN,evalX,evalY,batch_size):
    number_examples=len(evalX)
    eval_loss,eval_acc,eval_counter=0.0,0.0,0
    for start,end in zip(range(0,number_examples,batch_size),range(batch_size,number_examples,batch_size)):
        #this line does not call the train_op, so this line is to value but not train
        curr_eval_loss,curr_eval_acc= sess.run([CNN.loss,CNN.acc],#curr_eval_acc--->textCNN.accuracy
                                          feed_dict={CNN.input_x: evalX[start:end],CNN.input_y: evalY[start:end]})
        eval_loss,eval_acc,eval_counter=eval_loss+curr_eval_loss,eval_acc+curr_eval_acc,eval_counter+1
    return eval_loss/float(eval_counter),eval_acc/float(eval_counter)


def write_result(s):
    file_write = open(os.path.join(FLAGS.result_path, FLAGS.result_name), 'ab+')
    try:
        file_write.write(s + '\n')
    finally:
        file_write.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
