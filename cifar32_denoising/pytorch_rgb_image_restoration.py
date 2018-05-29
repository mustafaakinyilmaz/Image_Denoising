
"""
Created on Sat May 12 03:26:06 2018

@author: AkinYilmaz
"""


import matplotlib.pyplot as plt
import create_dataset as cr
import numpy as np
import time
from skimage.measure import compare_ssim,compare_psnr,compare_mse
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def compute_psnr(matris, mse, device):
    max_matris = matris.max()
    psnr = 20 * log10((max_matris/ (mse ** 0.5)), device)
    return psnr


def log10(x, device):
    numerator = torch.log(x)
    denominator = torch.log(torch.Tensor([10.0]).to(device).float())
    return numerator / denominator


def create_datasets():
    X_train, Y_train, X_test, Y_test = cr.create_dataset()

    Y_train = Y_train.reshape(-1, 3, 32, 32)
    X_train = X_train.reshape(-1, 3, 32, 32)

    Y_test = Y_test.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    return X_train.astype(np.float32), Y_train.astype(np.float32), X_test.astype(np.float32), Y_test.astype(np.float32)


def initialize_parameters(device):
    W1 = np.loadtxt("W1.txt").reshape(48, 3, 4, 4)
    W2 = np.loadtxt("W2.txt").reshape(96, 48, 4, 4)
    W3 = np.loadtxt("W3.txt").reshape(192, 96, 4, 4)


    parameters = {
        "W1": Variable(torch.from_numpy(W1).to(device).float(), requires_grad=True),
        "W2": Variable(torch.from_numpy(W2).to(device).float(), requires_grad=True),
        "W3": Variable(torch.from_numpy(W3).to(device).float(), requires_grad=True)
    }
    return parameters


def forward(X, parameters, device):
    X = torch.from_numpy(X).to(device).float()
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    conv1 = F.conv2d(X, W1, bias=None, padding=(1, 1), stride=2)
    relu1 = F.relu(conv1)

    conv2 = F.conv2d(relu1, W2, bias=None, padding=(1, 1), stride=2)
    relu2 = F.relu(conv2)

    conv3 = F.conv2d(relu2, W3, bias=None, padding=(1, 1), stride=2)
    relu3 = F.relu(conv3)

    unconv3 = F.conv_transpose2d(relu3, W3, bias=None, padding=(1, 1), stride=2)
    unrel3 = F.relu(unconv3)

    unconv2 = F.conv_transpose2d(unrel3, W2, bias=None, padding=(1, 1), stride=2)
    unrel2 = F.relu(unconv2)

    unconv1 = F.conv_transpose2d(unrel2, W1, bias=None, padding=(1, 1), stride=2)
    unconv1 = torch.clamp(unconv1, 0.0, 1.0)

    return unconv1


def compute_cost(out, Y, device):
    Y = torch.from_numpy(Y).to(device).float()
    cost = torch.mean((out - Y) ** 2)
    return cost


def model(device, X_train, Y_train, X_test, Y_test, num_epoch, learning_rate, batch_size):
    epoch_list = []
    train_cost_list = []
    test_cost_list = []
    #train_psnr_list = []
    #test_psnr_list = []

    total_train_batch = np.ceil(X_train.shape[0] / batch_size)
    total_test_batch = np.ceil(X_test.shape[0] / batch_size)

    parameters = initialize_parameters(device)

    optims = [parameters['W1'], parameters['W2'],parameters['W3']]

    optimizer = torch.optim.Adam(params=optims, lr=learning_rate, betas=(0.9, 0.99),eps=1.e-8)
    #optimizer = torch.optim.RMSprop(params=optims,lr=learning_rate,alpha=0.99,eps=1e-8)
    s_t = time.time()

    for epoch in range(1, num_epoch + 1):
        total_train_cost = 0
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

            optimizer.zero_grad()

            out = forward(minibatch_X, parameters, device)
            cost = compute_cost(out, minibatch_Y, device)

            cost.backward()

            optimizer.step()

            """print(parameters["W1"].grad[0,:,0,0])
            print("**")
            print(cost.item())"""


        for batch in range(0, X_train.shape[0], batch_size):
            if X_train.shape[0] - batch < batch_size:
                minibatch_train_X = X_train[batch:, :, :, :]
                minibatch_train_Y = Y_train[batch:, :, :, :]
            else:
                minibatch_train_X = X_train[batch:batch + batch_size, :, :, :]
                minibatch_train_Y = Y_train[batch:batch + batch_size, :, :, :]

            train_out = forward(minibatch_train_X, parameters, device)
            train_cost = compute_cost(train_out, minibatch_train_Y, device)
            #train_psnr = compute_psnr(train_out, train_cost, device)

            total_train_cost += train_cost.item() / total_train_batch
            #total_train_psnr += train_psnr.item() / total_train_batch

        for batch in range(0, X_test.shape[0], batch_size):
            if X_train.shape[0] - batch < batch_size:
                minibatch_test_X = X_test[batch:, :, :, :]
                minibatch_test_Y = Y_test[batch:, :, :, :]
            else:
                minibatch_test_X = X_test[batch:batch + batch_size, :, :, :]
                minibatch_test_Y = Y_test[batch:batch + batch_size, :, :, :]

            test_out = forward(minibatch_test_X, parameters, device)
            test_cost = compute_cost(test_out, minibatch_test_Y, device)
            #test_psnr = compute_psnr(test_out,test_cost,device)

            total_test_cost += test_cost.item() / total_test_batch
            #total_test_psnr += test_psnr.item() / total_test_batch


        epoch_list.append(epoch)
        train_cost_list.append(total_train_cost)
        test_cost_list.append(total_test_cost)
        #train_psnr_list.append(total_train_psnr)
        #test_psnr_list.append(total_test_psnr)
        print("epoch: " + str(epoch))
        print("training MSE: " + str(total_train_cost))
        #print("training PSNR: "+ str(total_train_psnr))
        print("***")
        print("test MSE: " + str(total_test_cost))
        #print("test PSNR: "+str(total_test_psnr))
        #print("time for 1 epoch: " + str(e_t - s_t) + " sec")
        print("-----------------")

    e_t = time.time()

    return epoch_list, train_cost_list, test_cost_list, test_out, minibatch_test_X, minibatch_test_Y,e_t-s_t


X_train,Y_train,X_test,Y_test = create_datasets()

device = torch.device("cuda:0")

#torch.backends.cudnn.deterministic = True


epoch_list,train_cost_list,test_cost_list,test_out,minibatch_test_X,minibatch_test_Y,train_time = model( device,
                                                                                                                                        X_train[:,:,:,:],
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



test_out_n = test_out.cpu().detach().numpy().astype(np.float32)


plt.imshow(test_out_n.reshape(-1,32,32,3)[50,:,:,:])
ssim_out = compare_ssim(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out_n.reshape(-1,32,32,3)[50,:,:,:],multichannel=True)
psnr_out = compare_psnr(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out_n.reshape(-1,32,32,3)[50,:,:,:])
mse_out = compare_mse(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],test_out_n.reshape(-1,32,32,3)[50,:,:,:])
plt.title("MSE: "+str(mse_out)+"\n"+
          "SSIM: "+str(ssim_out)+"\n"+
          "PSNR: "+str(psnr_out))


plt.imshow(((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[50,:,:,:])
ssim_x = compare_ssim(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[50,:,:,:],multichannel=True)
psnr_x = compare_psnr(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[50,:,:,:])
mse_x = compare_mse(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:],((minibatch_test_X*0.5)+0.5).reshape(-1,32,32,3)[50,:,:,:])
plt.title("MSE: "+str(mse_x)+"\n"+
          "SSIM: "+str(ssim_x)+"\n"+
          "PSNR: "+str(psnr_x))

plt.imshow(minibatch_test_Y.reshape(-1,32,32,3)[50,:,:,:])



16,50
"""plt.imshow(((X_train*0.5)+0.5).reshape(-1,128,32,3)[10,:,:,:])
plt.imshow(Y_train.reshape(-1,128,32,3)[10,:,:,:])

plt.imshow(((X_test*0.5)+0.5).reshape(-1,128,32,3)[1,:,:,:])
plt.imshow(Y_test.reshape(-1,128,32,3)[1,:,:,:])"""



np.mean()
