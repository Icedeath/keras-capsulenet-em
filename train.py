from keras.utils import to_categorical

def load_mnist(path):
    # the data, shuffled and split between train and test sets
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path)

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
    (x_train, y_train), (x_test, y_test) = load_mnist(args.data_dir)

    # define model
    model, eval_model, manipulate_model = CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)
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
