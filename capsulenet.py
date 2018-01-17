import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
#from utils import combine_images
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    """
    A Regular Layer: Tensor with shape [N, H, W, C], where often 
    N is the number of samples e.g., batch_size, 
    H is the height, 
    W is the width, and 
    C is the number of channels. 
    For example, in MNIST, each image is 28x28, with a single channel, so a batch size of 128 images would lead to an input tensor of [128, 28, 28, 1].    
    
    A Regular Kernel: Tensor with shape [KH, KW, I, O], where often 
    KH is the kernel height, 
    KW is the kernel width, 
    I is the number of input channels, and 
    O is the number of output channels, 
    e.g., a 5x5 convolution kernel on the previous [128, 28, 28, 1] to provide 32 ouptut channels would require a kernel shape [5, 5, 1, 32]. Of course, we also need strides to determine the height and width of the output layer.
    
    A Matrix Capsule Layer: Tensor tuple of (poses: [N, H, W, C, PH, PW], activations: [N, H, W, C]), where 
    PH is the pose height, and 
    PW is the pose width. 
    This is an extension from the regular layers to include poses and activations in representing a feature or an object. In the paper, matrix capsules with EM routing, PH = PW = 4.
    
    A Matrix Capsule Kernel: Tensor with shape [KH, KW, I, O, PH, PW]. 
    This kernel is used in the matrix capsules convolution operation to convert an inputs matrix capsule layer (poses: [N, H, W, I, PH, PW], activations: [N, H, W, I]) into and 
    output matrix capsule layer (poses: [N, OH, OW, O, PH, PW], activations: [N, OH, OW, O]). Here, OH is the output height, OW is the output width, and OH and OW are determined by the KH, KW and strides.
    
    In addition, I assume we use the MNIST dataset with a batchsize N=128, set A=32, B=48, C=64, D=80, and _E=10, so that we can distinguish between each layers.
    """

    # Layer 1: Just a conventional Conv2D layer
    # inputs [N, H, W, C] -> conv2d, 5x5, strides 2, channels 32 -> nets [N, OH, OW, 32]
    # Input Layer -> AL (A=32): kernel [5, 5, 1, 32], strides [1, 2, 2, 1], padding SAME, ReLU. This is a regular convoluation operation connects IL to AL.
    conv1 = layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu', name='conv1')(x)
    # Layer AL: [128, 14, 14, 32].


    # Layer 2:
    # AL -> BL (B=48): kernel [1, 1, 32, 48] x (4 x 4 + 1), strides [1, 1, 1, 1].
    # 16 such kernels of [1, 1, 32, 48] for building poses, and 1 such kernel [1, 1, 32, 48] for building activation. This is the initialization operation to connect
    # a regular layer to a matrix capsule layer, implemented in capsule_init() with details in next section.
    # inputs [N, H, W, C] -> conv2d, 1x1, strides 1, channels 32x(4x4+1) -> (poses, activations)
    poses, activations = PrimaryCap(conv1, matrix_dims_in_capsule=4, n_channels=32, kernel_size=1, strides=1, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    # digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
    #                          name='digitcaps')(primarycaps)
    #
    # # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # # If using tensorflow, this will not be necessary. :)
    # out_caps = Length(name='capsnet')(digitcaps)
    #
    # # Decoder network.
    # y = layers.Input(shape=(n_class,))
    # masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    # masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction
    #
    # # Shared Decoder model in training and prediction
    # decoder = models.Sequential(name='decoder')
    # decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    # decoder.add(layers.Dense(1024, activation='relu'))
    # decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    # decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    #
    # # Models for training and evaluation (prediction)
    # train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    # eval_model = models.Model(x, [out_caps, decoder(masked)])
    #
    # # manipulate model
    # noise = layers.Input(shape=(n_class, 16))
    # noised_digitcaps = layers.Add()([digitcaps, noise])
    # masked_noised_y = Mask()([noised_digitcaps, y])
    # manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    # return train_model, eval_model, manipulate_model



def load_mnist():
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Matrix Capsule with EM routing on MNIST or smallNORB.")
    parser.add_argument('--dataset', default=0, type=int, help='Dataset index to EM routing on : MNIST (0) or smallNORB (1)')
    parser.add_argument('--data_dir', default='data/', help='Specify a path for data directory')
    parser.add_argument('--log_dir', default='log/', help='Specify a path for log directory')
    parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--lr', default=0.001, type=float,
    #                     help="Initial learning rate")
    # parser.add_argument('--lr_decay', default=0.9, type=float,
    #                     help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    # parser.add_argument('--lam_recon', default=0.392, type=float,
    #                     help="The coefficient for the loss of decoder")
    # parser.add_argument('-r', '--routings', default=3, type=int,
    #                     help="Number of iterations used in routing algorithm. should > 0")
    # parser.add_argument('--shift_fraction', default=0.1, type=float,
    #                     help="Fraction of pixels to shift at most in each direction.")
    # parser.add_argument('--debug', action='store_true',
    #                     help="Save weights by TensorBoard")
    # parser.add_argument('--save_dir', default='./result')
    # parser.add_argument('-t', '--testing', action='store_true',
    #                     help="Test the trained model on testing dataset")
    # parser.add_argument('--digit', default=5, type=int,
    #                     help="Digit to manipulate")
    # parser.add_argument('-w', '--weights', default=None,
    #                     help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_mnist()

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=1)
    model.summary()

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        manipulate_latent(manipulate_model, (x_test, y_test), args)
        test(model=eval_model, data=(x_test, y_test), args=args)

