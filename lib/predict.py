from lib.Losses import cross_entropy
from lib.save import load_object
import numpy as np


def evaluate_and_predict(net, df_x, df_y):
    """
    Makes a prediction served to evaluate and/or predict
    :param net: neuronal network
    :param df_x: DataFrame containing xs
    :param df_y: DataFrame containing ys
    :return:
    """

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    tab_y_pred_softmax = np.array([[0, 0]])

    # predict
    if not net:
        net = load_object('network.save')
    out = net.predict(df_x)

    # formatage output data
    out = np.asarray(out).reshape(len(df_x), 2)
    y_list = list(df_y)
    tab_y_true_binary = np.asarray(y_list).reshape(len(y_list), 2)

    # calcul Prediction accuracy
    valid = 0
    index = 0
    for prediction in out:
        tab_y_pred_softmax = np.append(tab_y_pred_softmax, [prediction], axis=0)

        verification = tab_y_true_binary[index]
        if prediction[0] > prediction[1]:
            prediction[0] = 1
            prediction[1] = 0
        else:
            prediction[0] = 0
            prediction[1] = 1

        if np.array_equal(prediction, verification):
            valid += 1

        # Negatives
        if verification[0] == 0 and verification[1] == 1:
            if prediction[0] == 0 and prediction[1] == 1:
                TN += 1
            elif prediction[0] == 1 and prediction[1] == 0:
                FN += 1

        # Positives
        elif verification[0] == 1 and verification[1] == 0:
            if prediction[0] == 1 and prediction[1] == 0:
                TP += 1
            elif prediction[0] == 0 and prediction[1] == 1:
                FP += 1

        index += 1

    # del null row
    tab_y_pred_softmax = np.delete(tab_y_pred_softmax, 0, axis=0)

    print(f'\nPrediction evaluation of {index} elements :')
    print(f'True positive\t= {TP}')
    print(f'False positive\t= {FP}')
    print(f'True negative\t= {TN}')
    print(f'False negative\t= {FN}')
    print(f'Total\t\t= {TP + FP + TN + FN}\n')

    mse_accuracy = (TP + TN) / (TP + FN + FP + TN)
    positive_predictive_value = TP / (TP + FP)
    negative_predictive_value = TN / (FN + TN)

    print(f'MSE Accuracy\t\t= {mse_accuracy:.4f}')
    print(f'Positive Accuracy\t= {positive_predictive_value:.4f}')
    print(f'Negative Accuracy\t= {negative_predictive_value:.4f}')
    print(f'Cross_entropy Accuracy\t= {cross_entropy(tab_y_true_binary, tab_y_pred_softmax):.4f}')

    return mse_accuracy
