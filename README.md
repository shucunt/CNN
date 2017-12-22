# CNN
CNN with tensorflow with MNIST

input of train_CNN.py is a pickle file
pkl contain six data:trainX, validationX, testX, trainY, validationY, testY (only use train and validation)
all of those are list,and i translate them to np in train_CNN.py
i save the model in /workspace/checkpoint/CNN/ whose accuracy of validation is best

input of test_CNN.py is alse a pickle file
pkl contain six data:trainX, validationX, testX, trainY, validationY, testY (only use test)
and the output of of test_CNN.py is the average loss and accuracy of testX

CNN model with 2 conv layer and 2 max_pool, 1 fully connect layer and a classifier layer
both conv use a kernel with size 5,5 and stride is 1
conv1 has 32 filters, conv2 has 64 filters
both pooling use a kernel with 2,2 and stride is 2
before fc_layer, i reshape the output of pool2 layer
fully connect layer return a vector with 512 dimension

input        conv1_output  pooling1_output  conv2_output   pooling2_output      reshape      fc_output   num_class
1@28*28  ->  32@24*@4  ->  32@12*12  ->     64@8*8  ->     64@4*4           ->  64*4*4  ->   512  ->     10

after 20 epoch , the accuracy on test is 99.2%
