import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def load_training_data_simple(
    tracks_path = "../../data/training_set_preproc/log_mini.csv",
    sessions_path = "../../data/training_set_preproc/session_log_mini.csv"):

    tracks = pd.read_csv(tracks_path)
    sessions = pd.read_csv(sessions_path)

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

def load_test_data_simple(
    tracks_path = "../../data/test_set_preproc/log_mini.csv",
    sessions_path = "../../data/test_set_preproc/session_log_mini.csv"):

    tracks = pd.read_csv(tracks_path)
    sessions = pd.read_csv(sessions_path)

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

def save_submission(y_pred, session_lengths, path, overwrite = False):
    assert session_lengths.shape[0] == y_pred.shape[0]

    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    y_pred_01 = np.round((np.sign(y_pred) + 1) / 2).astype(int)

    write_type = 'a'
    if overwrite:
        write_type = 'w'

    with open(path, write_type) as f:
        for i in range(y_pred_01.shape[0]):
            ls = session_lengths[i]
            a = y_pred_01[i,int(ls/2):int(ls)]
            line = ' '.join(map(str,a))
            f.write(line + '\n')
    print('Written to file: %s' % path)
