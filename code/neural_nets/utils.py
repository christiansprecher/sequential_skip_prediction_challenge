import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


def load_training_data_simple():
    tracks = pd.read_csv("../../data/training_set_preproc/log_mini.csv")
    sessions = pd.read_csv("../../data/training_set_preproc/session_log_mini.csv")

    n = tracks.shape[0]
    s = sessions.shape[0]
    f1 = tracks.shape[1]
    f2 = sessions.shape[1]
    assert int(n/20) == s
    assert f1 == 42
    assert f2 == 11

    x_rnn = np.array(tracks.drop(['session_id','y2'], axis=1))
    x_rnn = np.reshape(x_rnn,(s,20,f1-2),order='C')

    x_fc = np.array(sessions.drop(['session_id'], axis=1))

    y = np.array(tracks['y2'])
    y = np.reshape(y,(s,20),order='C')

    return x_rnn, x_fc, y

def load_test_data_simple():
    tracks = pd.read_csv("../../data/test_set_preproc/log_mini.csv")
    sessions = pd.read_csv("../../data/test_set_preproc/session_log_mini.csv")

    n = tracks.shape[0]
    s = sessions.shape[0]
    f1 = tracks.shape[1]
    f2 = sessions.shape[1]
    assert int(n/20) == s
    assert f1 == 41
    assert f2 == 11

    x_rnn = np.array(tracks.drop(['session_id'], axis=1))
    x_rnn = np.reshape(x_rnn,(s,20,f1-1),order='C')

    x_fc = np.array(sessions.drop(['session_id'], axis=1))

    return x_rnn, x_fc
