import sys

import pandas as pd
import numpy as np

from lib.errors import display_errors_dict


def check_data_file(file):
    """
    :param file: The CSV file to check
    :return: Two DataFrames x and y
    """
    df = pd.read_csv(file, header=None)

    nb_column = len(list(df))

    if nb_column != 32:
        sys.exit(display_errors_dict('check_nb_column'))

    df_x = df.iloc[:, 2:].values.reshape(len(df), 1, len(list(df))-2)

    df_y = (df.iloc[:, 1].values == 'M').astype(int)
    test = []
    for i, val in enumerate(df_y):
        if val == 0:
            test.append([[0, 1]])
        else:
            test.append([[1, 0]])
    df_y = np.asarray(test)

    return df_x, df_y
