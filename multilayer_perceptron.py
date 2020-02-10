import argparse
import os
import sys

from lib.check_data import check_data_file
from lib.predict import evaluate_and_predict
from lib.errors import display_errors_dict

from lib.train import train


def parsing():
    """
    Parses and defines parameters
    :return: _args
    """

    parser = argparse.ArgumentParser(prog='py multilayer_perceptron.py')
    parser.add_argument('-t', '--train', help='Training mode "name_data_file.csv"', default=False)
    parser.add_argument('-p', '--predict', help='Predict mode "name_data_file.csv"', default=False)
    parser.add_argument('-e', '--epoch', help='number of epoch"', type=int, default=300)
    parser.add_argument('-r', '--learning', help='learning rate"', type=float, default=0.000045)
    parser.add_argument('-g', '--graphic', action='store_true', help='Graphic mode', default=False)
    parser.add_argument('-s', '--save', action='store_true', help='Save seed', default=False)
    parser.add_argument('-l', '--load', help='Load seed "name_seed.save"', default=False)

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parsing()

    # Check if train or predict is called
    if not args.train and not args.predict:
        sys.exit(display_errors_dict('no_order'))
    file_extension = '.csv'

    if args.train:
        file_name = args.train
    else:
        file_name = args.predict

    file = os.path.join(os.getcwd(), file_name)

    if os.path.exists(file):
        if os.path.isfile(file):
            if file.endswith(file_extension):
                df_x, df_y = check_data_file(file)
                if args.train:
                    train(args, df_x, df_y)
                else:
                    if os.path.exists(os.path.join(os.getcwd(), 'network.save')):
                        evaluate_and_predict(0, df_x, df_y)
                    else:
                        sys.exit(display_errors_dict('no_network'))
            else:
                sys.exit(display_errors_dict('endswith'))
        else:
            sys.exit(display_errors_dict('isfile'))
    else:
        sys.exit(display_errors_dict('exists'))
