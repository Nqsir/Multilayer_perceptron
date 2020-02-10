from lib.Network import Network
from lib.FCLayer import FCLayer
from lib.ActivationLayer import ActivationLayer
from lib.Activations import tanh, tanh_prime, softmax, softmax_prime
from lib.Losses import mse, mse_prime
from lib.errors import display_errors_dict

from sklearn.model_selection import train_test_split
from shutil import move
import matplotlib.pyplot as plt
import os
import sys

from lib.predict import evaluate_and_predict
from lib.save import save_object, load_object


def plot_loss(net):
    """
    Plots losses calculated from the training
    :param net: Network's instance
    """
    plt.style.use('ggplot')

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[7.5, 5])
    ax.set_title(f'Test', fontsize=14)
    ax.set_xlabel(f'Epochs')
    ax.set_ylabel(f'Loss / Val_loss')

    line1, = ax.plot(net.loss_list, 'c', zorder=1, label='loss')
    line2, = ax.plot(net.val_loss_list, 'r', zorder=1, label='val_loss')

    ax.grid(linestyle='-', linewidth=1)
    ax.legend(handles=[line1, line2], loc=1, fontsize=12)

    plt.tight_layout()
    plt.show()
    plt.clf()


def train(args, df_x, df_y):
    """
    :param args: dict from argparse
    :param df_x: DataFrame containing xs
    :param df_y: DataFrame containing ys
    :return: Network's instance
    """

    df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    net = Network()

    # Loads a seed if load
    if args.load:
        if os.path.exists(os.path.join(os.getcwd(), args.load)):
            net = load_object(args.load)
        else:
            sys.exit(display_errors_dict('wrong_load'))
    else:
        net.add(FCLayer(30, 30))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(30, 30))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(30, 30))
        net.add(ActivationLayer(tanh, tanh_prime))
        net.add(FCLayer(30, 2))
        net.add(ActivationLayer(softmax, softmax_prime))

    # Saves seed
    if args.save and not args.load:
        save_object(net, f'saved-seed.save')

    # Trains model
    net.use(mse, mse_prime)
    net.fit(df_x_train, df_y_train, df_x_test, df_y_test, epoch=args.epoch, learning_rate=args.learning, error_rate=0.01)
    plot_loss(net)

    # Evaluates with the 20% DataFrame
    rse_acc = evaluate_and_predict(net, df_x_test, df_y_test)

    # Renames seed with accuracy score
    if args.save and not args.load:
        move('saved-seed.save', f'seed-{int(rse_acc * 10000)}.save')

    # Saves network objet
    save_object(net, 'network.save')

    return net
