import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import os
import time
from joblib import Parallel, delayed
import multiprocessing

class Datagen:

    def __init__(self,folder_path):
        self.folder_path = folder_path

        if self.check_existance(folder_path):
            self.load_tracks()
        return

    def load_tracks(self):
        path = self.folder_path + "/track_features"
        path_output = path + "/pd_song.csv"

        # Check if output was already generated
        exists = self.check_existance(path_output)
        if exists:
            # Just load the file
            start = time.process_time()
            self.tracks = pd.read_csv(path_output)
            end = time.process_time()
            print("Tracks loaded")
            print("Time used: %4.2f seconds" % (end-start))

        else:
            # Merge two single files
            path_1 = path + "/tf_000000000000.csv"
            path_2 = path + "/tf_000000000001.csv"
            if not (self.check_existance(path_1) and self.check_existance(path_2)):
                return

            print("Merge two single track files: ")
            start = time.process_time()

            self.tracks = pd.read_csv(path_1).append(pd.read_csv(path_2))
            self.tracks.to_csv(path_output,index=False)

            end = time.process_time()
            print("Tracks merged")
            print("Time used: %4.2f seconds" % (end-start))

        return

    def load_training_data(self):
        path = self.folder_path + "/training_set"
        path_output = self.folder_path + "/training_set_preproc"

        if not os.path.exists(path_output):
            os.makedirs(path_output)

        all_training_files = glob.glob(path + "/*.csv")
        start = time.process_time()
        iter = 0
        num_files = len(all_training_files)

        print("Number of training files found: %u" % num_files)

        num_cores = min(multiprocessing.cpu_count(),8)
        Parallel(n_jobs=num_cores)(delayed(self.load_training_batch)(file,path_output) for file in all_training_files)

        # for file in all_training_files:
        #     iter = iter + 1
        #     self.load_training_batch(file,path_output)
        #
        #     end = time.process_time()
        #     print("Time used: %4.2f minutes for file %u/%u" % ((end-start)/60,iter,num_files))

    def load_training_batch(self,file,path_output,verbosity=True):
        print("Loading of file %s started" % os.path.basename(file))

        start = time.process_time()
        time_list = np.asarray([])

        # Load file
        batch = pd.read_csv(file).rename(columns={"track_id_clean": "track_id"})
        time_list = np.append(time_list,time.process_time())

        # Merge sessions and tracks
        batch_tmp = batch.copy()
        batch_tmp["Order"] = np.arange(len(batch_tmp))
        tmp = batch_tmp.merge(self.tracks, how='left', on="track_id").set_index("Order").iloc[np.arange(len(batch_tmp)), :]
        tmp.reset_index(drop=True)
        time_list = np.append(time_list,time.process_time())

        # Drop unwanted data
        track = tmp.drop(['session_position', 'session_length',
           'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
           'no_pause_before_play', 'short_pause_before_play',
           'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
           'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
           'hour_of_day', 'date', 'premium', 'context_type',
           'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end',
           'track_id'], axis=1)
        time_list = np.append(time_list,time.process_time())

        # Convert and normalize data
        #Do one-hot-encoding for mode and key
        track = pd.get_dummies(track, prefix=['key', 'mode'], columns=['key', 'mode']).drop(['key_11','mode_minor'],axis=1)

        #Normalize data
        tmp2 = track.drop(['session_id'],axis=1)
        t_min = tmp2.min()
        t_max = tmp2.max()

        tmp2_normal = (tmp2 - t_min) / (t_max - t_min)
        track = pd.DataFrame(track['session_id']).join(tmp2_normal)

        time_list = np.append(time_list,time.process_time())

        # Fill up sessions
        track_grouped = track.groupby("session_id")
        session_ids = batch["session_id"].drop_duplicates().reset_index(drop=True)
        n_rows = len(session_ids) * 20;

        data = np.transpose(np.array([np.zeros(n_rows, dtype=dt) for dt in track.dtypes]))
        sessions = pd.DataFrame(data)
        sessions.columns = track.columns

        for ix, item in session_ids.items():
            chk = track_grouped.get_group(item)
            L_s = len(chk)
            sessions.iloc[ix*20:(ix*20+L_s), :] = chk.values
            # sessions.iloc[ix*20+L_s:(ix+1)*20,"session_id"] = chk[0,"session_id"]
        time_list = np.append(time_list,time.process_time())

        # Create skip information and output vector
        k_y = tmp[["session_id", "skip_2"]]
        ky_grouped = k_y.groupby("session_id")
        kys = []
        for item in k_y["session_id"].drop_duplicates():
            chk = ky_grouped.get_group(item)
            L_s = len(chk)
            L_sh = int(L_s/2)
            ky_tmp = np.array(chk["skip_2"])*2-1
            ky_tmp = np.tile(ky_tmp.reshape((-1,1)), (1,2))
            ky_tmp[:L_sh,1] = 0
            ky_tmp[L_sh:,0] = 0
            kys.append(np.pad(ky_tmp, [(0,20-L_s),(0,0)], 'constant'))
        kys = np.array(kys)
        kys_shape = kys.shape
        kys = np.reshape(kys,(kys_shape[0]*kys_shape[1],kys_shape[2]))
        df_kys = pd.DataFrame(data=kys, columns=["y1", "y2"], dtype="int")
        result = pd.concat([sessions, df_kys], axis=1)
        time_list = np.append(time_list,time.process_time())

        #Save to csv file
        output_file_path = path_output + '/'+ os.path.basename(file)
        result.to_csv(output_file_path, index=False)
        time_list = np.append(time_list,time.process_time())

        # Print times
        time_list = (time_list - start) / 60;

        if verbosity:
            # print(result.head())
            print("Time [in minutes] used for loading batch: %4.2f, Merging tracks: %4.2f" % (time_list[0],time_list[1]),
            "Drop unwanted columns: %4.2f, Normalize data: %4.2f" % (time_list[2],time_list[3]),
            "Fill up sessions: %4.2f, Create skip information: %4.2f" % (time_list[4],time_list[5]),
            "Save to file %4.2f" % (time_list[6]))

        return

    def check_existance(self,path):
        exists = os.path.isfile(path) or os.path.exists(path)

        if not exists:
            print('File not found ' + path)
        return exists

    #
    #
    #
    #
    # def mergeLeftInOrder(x, y, on=None):
    #     x = x.copy()
    #     x["Order"] = np.arange(len(x))
    #     z = x.merge(y, how='left', on=on).set_index("Order").ix[np.arange(len(x)), :]
    #     return z.reset_index(drop=True)
    #
    #
    #
    # pd_playlist = pd_playlist.rename(columns={"track_id_clean": "track_id"})
    # tmp = mergeLeftInOrder(pd_playlist, pd_song, on="track_id")
    # track = tmp.drop(['session_position', 'session_length',
    #    'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
    #    'no_pause_before_play', 'short_pause_before_play',
    #    'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
    #    'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
    #    'hour_of_day', 'date', 'premium', 'context_type',
    #    'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end'], axis=1)
    # track_grouped = track.groupby("session_id")
    # sessions = pd.DataFrame(columns=track.columns)
    # for item in tqdm(pd_playlist["session_id"].drop_duplicates()):
    #     chk = track_grouped.get_group(item)
    #     L_s = len(chk)
    #     for i in range(20-L_s):
    #         s = pd.Series([0]*31, index=chk.columns, name='zero')
    #         chk = chk.append(s)
    #     sessions = sessions.append(chk.drop(["session_id", "track_id"], axis=1), ignore_index=True)
    #
    # k_y = tmp[["session_id", "skip_2"]]
    # ky_grouped = k_y.groupby("session_id")
    # kys = []
    # for item in tqdm(k_y["session_id"].drop_duplicates()):
    #     chk = ky_grouped.get_group(item)
    #     L_s = len(chk)
    #     L_sh = int(L_s/2)
    #     ky_tmp = np.array(chk["skip_2"])*2-1
    #     ky_tmp = np.tile(ky_tmp.reshape((-1,1)), (1,2))
    #     ky_tmp[:L_sh,1] = 0
    #     ky_tmp[L_sh:,0] = 0
    #     kys.append(np.pad(ky_tmp, [(0,20-L_s),(0,0)], 'constant'))
    # kys = np.array(kys)
    # df_kys = pd.DataFrame(data=kys, columns=["y1", "y2"], dtype="int")
    # #TODO append
    # result = pd.concat([sessions, df_kys], axis=1)
    #
    # return result
