# -*- coding: utf-8 -*-
#coding = utf-8
import tensorflow as tf

class MNIST_CNN:
    def __init__(self, num_class, learning_rate, batch_size, decay_steps, decay_rate, l2_lambda, input_size, num_channels,
        conv1_filter, kernel1_size, conv2_filter, kernel2_size, fc_size, dropout_keep_prob, train, 
        initializer = tf.random_normal_initializer(stddev=0.1)):

        self.num_class = num_class
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.l2_lambda = l2_lambda
        self.input_size = input_size
        self.num_channels = num_channels
        self.conv1_filter = conv1_filter
        self.kernel1_size = kernel1_size
        self.conv2_filter = conv2_filter
        self.kernel2_size = kernel2_size
        self.fc_size = fc_size
        self.dropout_keep_prob = dropout_keep_prob
        self.train = train
        self.initializer = initializer


        self.input_x = tf.placeholder(tf.float32, [None, self.input_size], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')


        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.epoch_step = tf.Variable(0, trainable=False, name='epoch_step')
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.output_cnn = self.inference()
        self.logits = self.classifier()
        self.loss = self.loss()
        self.prediction = tf.argmax(self.logits, axis = 1, name='prediction')
        correct_prediction = tf.equal(tf.cast(self.prediction, tf.int32), self.input_y)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='acc')
        if not self.train:
            return 
        self.train_op = self.train_op()



    def inference(self):
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                "weight", [self.kernel1_size, self.kernel1_size, self.num_channels, self.conv1_filter],
                initializer=self.initializer)
            conv1_biases = tf.get_variable("bias", [self.conv1_filter], initializer=self.initializer)
            conv1 = tf.nn.conv2d(tf.reshape(self.input_x, [-1, 28 , 28,1]), conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))  #24 * 24 * 32
            # print 'relu1'
            # print relu1

        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")  # 12 * 12 * 32
            # print 'pool1'
            # print pool1

        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable(
                "weight", [self.kernel2_size, self.kernel2_size, self.conv1_filter, self.conv2_filter],
                initializer=self.initializer)
            conv2_biases = tf.get_variable("bias", [self.conv2_filter], initializer=self.initializer)
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases)) #8 * 8 64
            # print 'relu2'
            # print relu2

        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]  #4 * 4 * 64
            # print pool_shape
            # print pool2
            reshaped = tf.reshape(pool2, [-1, nodes])

        with tf.variable_scope('layer5-fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, self.fc_size],
                                          initializer=self.initializer)
            fc1_biases = tf.get_variable("bias", [self.fc_size], initializer=self.initializer)

            output_cnn = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if self.train: output_cnn = tf.nn.dropout(output_cnn, self.dropout_keep_prob)
        return output_cnn

    def classifier(self):
        with tf.variable_scope('classifier'):
            fc2_weights = tf.get_variable("weight", [self.fc_size, self.num_class],
                                          initializer=self.initializer)
            fc2_biases = tf.get_variable("bias", [self.num_class], initializer=self.initializer)
            
            logit = tf.matmul(self.output_cnn, fc2_weights) + fc2_biases
        return logit

    def loss(self):
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.l2_lambda
            loss = loss + l2_losses
        return loss

    def train_op(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step, learning_rate = learning_rate, optimizer="Adam")
        return train_op

