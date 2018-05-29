

def create_dataset():
    import numpy as np
    import glob
    import matplotlib.image as matim
    np.random.seed(12)
    all_images = glob.glob("cifar32/*.png")

    array = []

    for images in all_images:
        array_im = matim.imread(images)
        array.append(array_im)

    image_array = np.array(array)
    print(image_array.shape)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(image_array, test_size=0.2)
    """train = image_array[:20000,:,:,:]
    test = image_array[20000:25000,:,:,:]"""

    X_train = (train - 0.5)/0.5
    Y_train = train

    X_test = (test - 0.5)/0.5
    Y_test = test

    variances = np.var(X_train, axis=(0, 1, 2))

    mean = 0
    var = variances / 2.0
    std = var**0.5
    print(std)

    X_train[:, :, :, 0] = np.clip(X_train[:,:,:,0] + np.random.normal(mean,std[0],size=(X_train.shape[0],32,32)),-1.0,1.0)
    X_train[:, :, :, 1] = np.clip(X_train[:, :, :, 1] + np.random.normal(mean, std[1], size=(X_train.shape[0], 32, 32)),-1.0,1.0)
    X_train[:, :, :, 2] = np.clip(X_train[:, :, :, 2] + np.random.normal(mean, std[2], size=(X_train.shape[0], 32, 32)),-1.0,1.0)

    X_test[:, :, :, 0] = np.clip(X_test[:, :, :, 0] + np.random.normal(mean, std[0], size=(X_test.shape[0], 32, 32)),-1.0,1.0)
    X_test[:, :, :, 1] = np.clip(X_test[:, :, :, 1] + np.random.normal(mean, std[1], size=(X_test.shape[0], 32, 32)),-1.0,1.0)
    X_test[:, :, :, 2] = np.clip(X_test[:, :, :, 2] + np.random.normal(mean, std[2], size=(X_test.shape[0], 32, 32)),-1.0,1.0)

    print(np.var(X_test,axis=(0,1,2)))
    #print("%.8f"%X_test[0,0,0,0])
    print(X_test[0,0,0,0])
    return X_train,Y_train,X_test,Y_test