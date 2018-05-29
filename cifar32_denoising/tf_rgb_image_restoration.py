
"""
Created on Sat May 23 21:26:06 2018

@author: AkinYilmaz
"""



import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
import create_dataset as cr
from skimage.measure import compare_psnr,compare_ssim,compare_mse

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"




def create_placeholders(n_h,n_w,n_c):
    X = tf.placeholder(dtype=tf.float32,shape= [None,n_h,n_w,n_c], name="X")
    Y = tf.placeholder(dtype=tf.float32,shape= [None,n_h,n_w,n_c], name="Y")
    return X,Y

def compute_psnr(matris,mse):
    max_matris = tf.reduce_max(matris)
    psnr = 20*log10(max_matris/tf.sqrt(mse))
    return psnr

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10,dtype=numerator.dtype))
    return numerator/denominator


def create_datasets():
    X_train, Y_train, X_test, Y_test = cr.create_dataset()

    Y_train = Y_train.reshape(-1, 32, 32, 3)
    X_train = X_train.reshape(-1, 32, 32, 3)

    Y_test = Y_test.reshape(-1, 32, 32, 3)
    X_test = X_test.reshape(-1, 32, 32, 3)

    return X_train.astype(np.float32), Y_train.astype(np.float32), X_test.astype(np.float32), Y_test.astype(np.float32)




def initialize_parameters():

    W1 = np.loadtxt("W1.txt").reshape(4, 4, 3, 48)
    W2 = np.loadtxt("W2.txt").reshape(4, 4, 48, 96)
    W3 = np.loadtxt("W3.txt").reshape(4, 4, 96, 192)

    parameters = {
        "W1": tf.Variable(W1, name="W1", trainable=True, dtype=tf.float32),
        "W2": tf.Variable(W2, name="W2", trainable=True, dtype=tf.float32),
        "W3": tf.Variable(W3, name="W3", trainable=True, dtype=tf.float32)

    }
    return parameters


def forward(X, parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    #paddings = tf.constant([[0,0],[1,1],[1,1],[0,0]])

    #X = tf.pad(X,paddings=paddings,mode="CONSTANT")
    conv1 = tf.nn.conv2d(X, W1, strides=[1, 2, 2, 1],padding="SAME")
    relu1 = tf.nn.relu(conv1)

    #relu1 = tf.pad(relu1,paddings=paddings,mode="CONSTANT")
    conv2 = tf.nn.conv2d(relu1, W2, strides=[1, 2, 2, 1], padding="SAME")
    relu2 = tf.nn.relu(conv2)

    #relu2 = tf.pad(relu2,paddings=paddings,mode="CONSTANT")
    conv3 = tf.nn.conv2d(relu2, W3, strides=[1, 2, 2, 1], padding="SAME")
    relu3 = tf.nn.relu(conv3)

    #conv3 = tf.pad(conv3,paddings=paddings,mode="CONSTANT")
    unconv3 = tf.nn.conv2d_transpose(relu3, W3, strides=[1, 2, 2, 1], output_shape=[tf.shape(X)[0], 8, 8, 96],padding="SAME")
    unrel3 = tf.nn.relu(unconv3)

    unconv2 = tf.nn.conv2d_transpose(unrel3, W2, strides=[1, 2, 2, 1], output_shape=[tf.shape(X)[0], 16, 16, 48],padding="SAME")
    unrel2 = tf.nn.relu(unconv2)

    unconv1 = tf.nn.conv2d_transpose(unrel2, W1, strides=[1, 2, 2, 1], output_shape=[tf.shape(X)[0], 32, 32, 3],padding="SAME")
    unconv1 = tf.clip_by_value(unconv1, 0.0, 1.0)

    return unconv1


def compute_cost(out, Y):
    cost = tf.reduce_mean(tf.pow(out - Y, 2))
    return cost


def model(X_train, Y_train, X_test, Y_test, num_epoch, learning_rate, batch_size):
    (m, n_h, n_w, n_c) = X_train.shape

    epoch_list = []
    train_cost_list = []
    test_cost_list = []
    #train_psnr_list = []
    #test_psnr_list = []

    total_batch = np.ceil(X_train.shape[0] / batch_size)
    total_test_batch = np.ceil(X_test.shape[0] / batch_size)

    device_name = "/gpu:0"
    with tf.device(device_name):
        print("using gpu")
        tf.reset_default_graph()

        X, Y = create_placeholders(n_h, n_w, n_c)

        parameters = initialize_parameters()

        out = forward(X, parameters)
        cost = compute_cost(out, Y)
        #psnr = compute_psnr(out, cost)

        optims = [parameters["W1"], parameters["W2"], parameters["W3"]]

        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.99, epsilon=1.e-8).minimize(cost,
                                                                                                var_list=optims)
        #W1_grad = tf.gradients(cost,[parameters["W1"]])[0]
        init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        sess.run(init)
        start_time = time.time()
        for epoch in range(1, num_epoch + 1):
            total_cost = 0
            total_test_cost = 0
            #total_train_psnr = 0
            #total_test_psnr = 0
            for batch in range(0, X_train.shape[0], batch_size):
                if X_train.shape[0] - batch < batch_size:
                    minibatch_X = X_train[batch:, :, :, :]
                    minibatch_Y = Y_train[batch:, :, :, :]
                else:
                    minibatch_X = X_train[batch:batch + batch_size, :, :, :]
                    minibatch_Y = Y_train[batch:batch + batch_size, :, :, :]

                #print(sess.run(cost, feed_dict={X: minibatch_X, Y: minibatch_Y}))


                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y})
                """grad = sess.run(W1_grad, feed_dict={X: minibatch_X, Y: minibatch_Y})
                print(grad[0, 0, :, 0])
                print("******")"""


            for batch in range(0, X_train.shape[0], batch_size):
                if X_train.shape[0] - batch < batch_size:
                    minibatch_X = X_train[batch:, :, :, :]
                    minibatch_Y = Y_train[batch:, :, :, :]
                else:
                    minibatch_X = X_train[batch:batch + batch_size, :, :, :]
                    minibatch_Y = Y_train[batch:batch + batch_size, :, :, :]

                batch_cost = sess.run(cost, feed_dict={X: minibatch_X, Y: minibatch_Y})
                total_cost += batch_cost / total_batch

                #train_psnr = sess.run(psnr, feed_dict={X: minibatch_X, Y: minibatch_Y})
                #total_train_psnr += train_psnr / total_batch

            for batch in range(0, X_test.shape[0], batch_size):
                if X_test.shape[0] - batch < batch_size:
                    minibatch_test_X = X_test[batch:, :, :, :]
                    minibatch_test_Y = Y_test[batch:, :, :, :]
                else:
                    minibatch_test_X = X_test[batch:batch + batch_size, :, :, :]
                    minibatch_test_Y = Y_test[batch:batch + batch_size, :, :, :]

                test_out, test_batch_cost = sess.run([out, cost], feed_dict={X: minibatch_test_X, Y: minibatch_test_Y})
                total_test_cost += test_batch_cost / total_test_batch

                #test_psnr = sess.run(psnr, feed_dict={X: minibatch_test_X, Y: minibatch_test_Y})
                #total_test_psnr += test_psnr / total_test_batch

            #train_psnr_list.append(total_train_psnr)
            #test_psnr_list.append(total_test_psnr)
            epoch_list.append(epoch)
            train_cost_list.append(total_cost)
            test_cost_list.append(total_test_cost)
            print("epoch: " + str(epoch))
            print("training cost: " + str(total_cost))
            #print("training psnr: " + str(total_train_psnr))
            print("***")
            print("test cost: " + str(total_test_cost))
            #print("test psnr: " + str(total_test_psnr))
            print("-----------------")


        end_time = time.time()

    return epoch_list,train_cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,end_time-start_time


X_train,Y_train,X_test,Y_test = create_datasets()


epoch_list,train_cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_time = model(X_train[:,:,:,:],
                                                                                                                                       Y_train[:,:,:,:],
                                                                                                                                       X_test[:,:,:,:],
                                                                                                                                       Y_test[:,:,:,:],
                                                                                                                                       num_epoch=25,
                                                                                                                                       learning_rate=3.e-3,
                                                                                                                                       batch_size=256)


plt.plot(epoch_list,train_cost_list,'-b',label="training MSE")
plt.plot(epoch_list,test_cost_list,'-r',label="test MSE")
plt.xlabel("# of iterations")
plt.ylabel("MSE")
plt.title("training time: "+str(train_time)+" secs"+"\n"+
          "training MSE after 5 iterations: "+str(train_cost_list[4])+"\n"+
          "training MSE after 10 iterations: "+str(train_cost_list[9])+"\n"+
          "training MSE after 15 iterations: "+str(train_cost_list[14])+"\n"+
          "training MSE after 20 iterations: "+str(train_cost_list[19])+"\n"+
          "training MSE after 25 iterations: "+str(train_cost_list[24]))
plt.grid()
plt.legend()

"""plt.plot(epoch_list,train_psnr_list,'-b',label="training PSNR")
plt.plot(epoch_list,test_psnr_list,'-r',label="test PSNR")
plt.title("Final training PSNR: "+str(train_psnr_list[-1])+"\n"+
          "Final test PSNR: " + str(test_psnr_list[-1]))
plt.xlabel("# of iterations")
plt.ylabel("PSNR")
plt.grid()
plt.legend()"""


plt.imshow(test_out.reshape(-1,32,32,3)[50,:,:,:])
ssim_out = compare_ssim(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out.reshape(-1,32,32,3)[50,:,:,:],multichannel=True)
psnr_out = compare_psnr(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out.reshape(-1,32,32,3)[50,:,:,:])
mse_out = compare_mse(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out.reshape(-1,32,32,3)[50,:,:,:])
plt.title("MSE: "+str(mse_out)+"\n"+
          "SSIM: "+str(ssim_out)+"\n"+
          "PSNR: "+str(psnr_out))


plt.imshow(((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[16,:,:,:])
ssim_x = compare_ssim(minibatch_test_Y.reshape(-1,32,32,3)[16,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[16,:,:,:],multichannel=True)
psnr_x = compare_psnr(minibatch_test_Y.reshape(-1,32,32,3)[16,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[16,:,:,:])
mse_x = compare_mse(minibatch_test_Y.reshape(-1,32,32,3)[16,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[16,:,:,:])
plt.title("MSE: "+str(mse_x)+"\n"+
          "SSIM: "+str(ssim_x)+"\n"+
          "PSNR: "+str(psnr_x))

plt.imshow(minibatch_test_Y.reshape(-1,32,32,3)[16,:,:,:])

#plt.imshow(((X_train*0.5)+0.5).reshape(-1,32,32,3)[10,:,:,:])
16,34
"""tf.reset_default_graph()
X, Y = create_placeholders(32, 32, 3)
parameters = initialize_parameters()
out = forward(X, parameters)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    output = sess.run(out,feed_dict={X:X_train[0:100,:,:,:],Y:Y_train[0:100,:,:,:]})


plt.imshow((X_train+0.5).reshape(-1,128,32,3)[3,:,:,:])
plt.imshow(Y_train.reshape(-1,128,32,3)[3,:,:,:])

plt.imshow((X_test+0.5).reshape(-1,128,32,3)[1,:,:,:])
plt.imshow(Y_test.reshape(-1,96,32,3)[1,:,:,:])

plt.imshow(output.reshape(-1,64,32,3)[1,:,:,:])"""
