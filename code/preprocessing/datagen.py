import pandas as pd
import numpy as np
import glob
import os
import time
from joblib import Parallel, delayed
import multiprocessing

class Datagen:

    def __init__(self,folder_path,cores, overwrite=True, reverse = False, verbosity=True):
        self.folder_path = folder_path
        self.verbosity = int(verbosity)
        self.overwrite = int(overwrite)
        self.cores = int(cores)
        self.reverse = int(reverse)

        if self.check_existance(folder_path):
            self.load_tracks()
            self.create_feature_limits()
            self.add_all_categories()
        return

    def load_tracks(self):

        print("Loading tracks")
        path = self.folder_path + "/track_features"
        path_output = path + "/pd_song.csv"

        # Check if output was already generated
        exists = self.check_existance(path_output)
        if exists:
            # Just load the file
            start = time.process_time()
            self.tracks = pd.read_csv(path_output)
            end = time.process_time()
            print("Tracks loaded, time used: %4.2f seconds" % (end-start))

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
            print("Tracks merged, time used: %4.2f seconds" % (end-start))

        return

    def create_feature_limits(self):
        # Define static bounds for normalization
        track_min = {'duration' : 30, 'release_year' : 1950, 'us_popularity_estimate' : 90,
            'acousticness' : 0, 'beat_strength' : 0, 'bounciness' : 0,
            'danceability' : 0, 'dyn_range_mean' : 0, 'energy' : 0, 'flatness' : 0,
            'instrumentalness' : 0, 'liveness' : 0, 'loudness' : -60,
            'mechanism' : 0, 'organism' : 0, 'speechiness' : 0, 'tempo' : 0, 'time_signature' : 0,
            'valence' : 0, 'acoustic_vector_0' : -1, 'acoustic_vector_1' : -1,
            'acoustic_vector_2' : -1, 'acoustic_vector_3' : -1, 'acoustic_vector_4' : -1,
            'acoustic_vector_5': -1, 'acoustic_vector_6' : -1, 'acoustic_vector_7' : -1,
            'key_0' : 0, 'key_1' : 0, 'key_2' : 0, 'key_3' : 0, 'key_4' : 0, 'key_5' : 0,
            'key_6' : 0, 'key_7' : 0, 'key_8' : 0, 'key_9' : 0, 'key_10' : 0, 'mode_major' : 0}
        track_max = {'duration' : 1800, 'release_year' : 2018, 'us_popularity_estimate' : 100,
            'acousticness' : 1, 'beat_strength' : 1, 'bounciness' : 1,
            'danceability' : 1, 'dyn_range_mean' : 40, 'energy' : 1, 'flatness' : 1,
            'instrumentalness' : 1, 'liveness' : 1, 'loudness' : 4,
            'mechanism' : 1, 'organism' : 1, 'speechiness' : 1, 'tempo' : 244, 'time_signature' : 5,
            'valence' : 1, 'acoustic_vector_0' : 1, 'acoustic_vector_1' : 1,
            'acoustic_vector_2' : 1, 'acoustic_vector_3' : 1, 'acoustic_vector_4' : 1,
            'acoustic_vector_5': 1, 'acoustic_vector_6' : 1, 'acoustic_vector_7' : 1,
            'key_0' : 1, 'key_1' : 1, 'key_2' : 1, 'key_3' : 1, 'key_4' : 1, 'key_5' : 1,
            'key_6' : 1, 'key_7' : 1, 'key_8' : 1, 'key_9' : 1, 'key_10' : 1, 'mode_major' : 1}
        columns = ['duration', 'release_year', 'us_popularity_estimate', 'acousticness',
            'beat_strength', 'bounciness', 'danceability', 'dyn_range_mean',
            'energy', 'flatness', 'instrumentalness', 'liveness', 'loudness',
            'mechanism', 'organism', 'speechiness', 'tempo', 'time_signature',
            'valence', 'acoustic_vector_0', 'acoustic_vector_1',
            'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
            'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7', 'key_0',
            'key_1', 'key_2', 'key_3', 'key_4', 'key_5', 'key_6', 'key_7', 'key_8',
            'key_9', 'key_10', 'mode_major']

        self.t_min = pd.Series(track_min).reindex(index=columns)
        self.t_max = pd.Series(track_max).reindex(index=columns)


        session_min = {'session_length' : 10, 'hour_of_day' : 0, 'date' : 1, 'shuffle_True' : 0,
            'premium_True' : 0, 'context_catalog' : 0, 'context_charts' : 0,
            'context_editorial_playlist' : 0, 'context_personalized_playlist' : 0,
            'context_radio' : 0}
        session_max = {'session_length' : 20, 'hour_of_day' : 23, 'date' : 365, 'shuffle_True' : 1,
            'premium_True' : 1, 'context_catalog' : 1, 'context_charts' : 1,
            'context_editorial_playlist' : 1, 'context_personalized_playlist' : 1,
            'context_radio' : 1}
        columns = ['session_length', 'hour_of_day', 'date', 'shuffle_True', 'premium_True',
               'context_catalog', 'context_charts', 'context_editorial_playlist',
               'context_personalized_playlist', 'context_radio']

        self.s_min = pd.Series(session_min).reindex(index=columns)
        self.s_max = pd.Series(session_max).reindex(index=columns)
        return

    def add_all_categories(self):
        columns_t = ['session_id', 'duration', 'release_year', 'us_popularity_estimate',
            'acousticness', 'beat_strength', 'bounciness', 'danceability',
            'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
            'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness',
            'tempo', 'time_signature', 'valence', 'acoustic_vector_0',
            'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
            'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6',
            'acoustic_vector_7']
        series = pd.Series([0]*30)
        series.index = columns_t
        series["mode"] = 'major'

        track_extended = pd.concat([series] * 13, axis = 1).transpose()
        for i in range(-1,12,1):
            track_extended.at[i+1,"key"] = i

        # track_extended["mode"].iloc[0] = 'minor'
        track_extended.at[0,"mode"] = 'minor'

        self.track_extended = track_extended

        columns_s = ['session_id', 'session_length', 'hist_user_behavior_is_shuffle',
            'hour_of_day', 'date', 'premium', 'context_type']
        series = pd.Series([0]*7)
        series.index = columns_s
        series["hist_user_behavior_is_shuffle"] = True
        series["premium"] = True
        series["context_type"] = 'editorial_playlist'

        session_extended = pd.concat([series] * 5, axis = 1).transpose()
        session_extended.at[1, "context_type"] = 'charts'
        session_extended.at[2, "context_type"] = 'catalog'
        session_extended.at[3, "context_type"] = 'radio'
        session_extended.at[4, "context_type"] = 'user_collection'
        session_extended.at[1, "hist_user_behavior_is_shuffle"] = False
        session_extended.at[1, "premium"] = False

        self.session_extended = session_extended

        return

    def load_training_data(self):
        path = self.folder_path + "/training_set"
        path_output = self.folder_path + "/training_set_preproc"

        if not os.path.exists(path_output):
            os.makedirs(path_output)

        all_training_files = glob.glob(path + "/*.csv")
        start = time.process_time()

        num_files = len(all_training_files)
        print("Number of training files found: %u" % num_files)

        num_cores = min(multiprocessing.cpu_count(),8, self.cores)

        if(self.reverse):
            Parallel(n_jobs=num_cores)(delayed(self.load_training_batch)(file,path_output)
                for file in reversed(all_training_files))
        else:
            Parallel(n_jobs=num_cores)(delayed(self.load_training_batch)(file,path_output)
                for file in all_training_files)

        end = time.process_time()
        print("Time used for processing all training files: %4.2f" % ((end-start)/60))
        return

    def load_training_batch(self,file,path_output):
        print("Loading of file %s started" % os.path.basename(file))

        output_file_path = path_output + '/'+ os.path.basename(file)

        if self.check_existance(output_file_path,False) and not self.overwrite:
            print("Output file %s already exists" % os.path.basename(file))
            return

        start = time.process_time()
        time_list = np.asarray([])

        # Load file
        batch = pd.read_csv(file).rename(columns={"track_id_clean": "track_id"})
        time_list = np.append(time_list,time.process_time())

        # Start with item based information
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

        del batch_tmp
        # Convert and normalize data
            #Do one-hot-encoding for mode and key
            #Add first extended state to feature all add_all_categories

        track_ext  = pd.concat([track,self.track_extended], axis = 0)
        track_ext = pd.get_dummies(track_ext, prefix=['key', 'mode'], columns=['key', 'mode'])

        n = self.track_extended.shape[0]
        track = track_ext[:-n].drop(['key_-1','key_11','mode_minor'],axis=1)

        del track_ext
            #Normalize data
        tmp2 = track.drop(['session_id'],axis=1)
        tmp2_normal = (tmp2 - self.t_min) / (self.t_max - self.t_min)
        track = pd.DataFrame(track['session_id']).join(tmp2_normal)
        time_list = np.append(time_list,time.process_time())

        # Fill up sessions
        track_grouped = track.groupby("session_id")
        session_ids = batch["session_id"].drop_duplicates().reset_index(drop=True)
        n_rows = len(session_ids) * 20;

        data = np.transpose(np.array([np.zeros(n_rows, dtype=dt) for dt in track.dtypes]))
        sessions = pd.DataFrame(data)
        sessions.columns = track.columns

        del track, data

        for ix, item in session_ids.items():
            chk = track_grouped.get_group(item)
            L_s = len(chk)
            sessions.iloc[ix*20:(ix*20+L_s), :] = chk.values
            # sessions.iloc[ix*20+L_s:(ix+1)*20,"session_id"] = chk[0,"session_id"]
        time_list = np.append(time_list,time.process_time())

        del track_grouped

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

        del kys, k_y, ky_grouped, df_kys

        #create sessions_based_information
        #Remove unwanted information
        session_fixed = batch.drop(['session_position', 'track_id',
           'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
           'no_pause_before_play', 'short_pause_before_play',
           'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
           'hist_user_behavior_n_seekback',
           'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end'], axis=1)
        session_fixed_ids = session_fixed["session_id"].drop_duplicates().reset_index(drop=True)
        session_grouped = session_fixed.groupby("session_id")

        del batch
        #Create empty variable to store values in
        n_rows = len(session_fixed_ids);
        data = np.transpose(np.array([np.empty(n_rows, dtype=dt) for dt in session_fixed.dtypes]))
        session_single = pd.DataFrame(data)
        session_single.columns = session_fixed.columns

        del session_fixed, data

        #Fill in session information with information of last element of first half
        for ix, item in session_fixed_ids.items():
            chk = session_grouped.get_group(item)
            L_s = len(chk)
            L_sh = int(L_s/2) - 1
            session_single.iloc[ix] = chk.iloc[L_sh]

        del session_fixed_ids, session_grouped

        #Modify date and perform one-hot-encoding
        session_single['date'] = pd.to_datetime(session_single['date'],
            errors='coerce').apply(lambda x: x.dayofyear)

        session_ext  = pd.concat([session_single,self.session_extended], axis = 0)
        session_ext = pd.get_dummies(session_ext, prefix=['shuffle', 'premium', 'context'],
            columns=['hist_user_behavior_is_shuffle', 'premium','context_type'])

        n = self.session_extended.shape[0]
        session_single = session_ext[:-n].drop(['session_id','shuffle_False',
            'premium_False','context_user_collection'],axis=1)

        del session_ext

        #Normalize data
        tmp3 = (session_single - self.s_min) / (self.s_max - self.s_min)
        session_finish = pd.DataFrame(session_ids).join(tmp3)

        del tmp3

        #Save to csv file
        print("Start saving to file %s" % os.path.basename(file))
        output_file_path = path_output + '/'+ os.path.basename(file)
        result.to_csv(output_file_path, index=False)
        time_list = np.append(time_list,time.process_time())

        #Save to csv file
        output_file_path = path_output + '/session_'+ os.path.basename(file)
        session_finish.to_csv(output_file_path, index=False)
        time_list = np.append(time_list,time.process_time())

        print("Finished saving to file %s" % os.path.basename(file))

        # Print times
        time_list = (time_list - start) / 60;
        if self.verbosity:
            # print(result.head())
            print("Loading of file %s finished" % os.path.basename(file))
            print("Time [in minutes] used for loading batch: %4.2f, Merging tracks: %4.2f" % (time_list[0],time_list[1]),
            "Drop unwanted columns: %4.2f, Normalize data: %4.2f," % (time_list[2],time_list[3]),
            "Fill up sessions: %4.2f, Create skip information: %4.2f," % (time_list[4],time_list[5]),
            "Save to file: %4.2f, Create Session fixed information: %4.2f" % (time_list[6],time_list[7]))

        return


    def load_test_data(self):
        path = self.folder_path + "/test_set"
        path_output = self.folder_path + "/test_set_preproc"

        if not os.path.exists(path_output):
            os.makedirs(path_output)

        all_test_files = glob.glob(path + "/*input*.csv")
        start = time.process_time()

        num_files = len(all_test_files)
        print("Number of test files found: %u" % num_files)

        num_cores = min(multiprocessing.cpu_count(),8,self.cores)

        if(self.reverse):
            Parallel(n_jobs=num_cores)(delayed(self.load_test_batch)(file,path_output)
                for file in reversed(all_test_files))
        else:
            Parallel(n_jobs=num_cores)(delayed(self.load_test_batch)(file,path_output)
                for file in all_test_files)

        end = time.process_time()
        print("Time used for processing all test files: %4.2f minutes" % ((end-start)/60))
        return

    def load_test_batch(self,file,path_output):
        print("Loading of file %s started" % os.path.basename(file))

        output_file_path = path_output + '/' + os.path.basename(file)
        output_file_path = output_file_path.replace('input_','')

        if self.check_existance(output_file_path,False) and not self.overwrite:
            print("Output file %s already exists" % os.path.basename(file))
            return

        start = time.process_time()
        time_list = np.asarray([])

        #load files
        file_ph = file.replace('input','prehistory')
        log_ip = pd.read_csv(file).rename(columns={"track_id_clean": "track_id"})
        log_ph = pd.read_csv(file_ph).rename(columns={"track_id_clean": "track_id"})
        time_list = np.append(time_list,time.process_time())

        # Start with item based information
        # Merge sessions and tracks
        log_ip_tmp = log_ip.copy()
        log_ip_tmp["Order"] = np.arange(len(log_ip_tmp))
        tmp_ip = log_ip_tmp.merge(self.tracks, how='left', on="track_id").set_index("Order").iloc[np.arange(len(log_ip_tmp)), :]
        tmp_ip.reset_index(drop=True)

        log_ph_tmp = log_ph.copy()
        log_ph_tmp["Order"] = np.arange(len(log_ph_tmp))
        tmp_ph = log_ph_tmp.merge(self.tracks, how='left', on="track_id").set_index("Order").iloc[np.arange(len(log_ph_tmp)), :]
        tmp_ph.reset_index(drop=True)
        time_list = np.append(time_list,time.process_time())

        # Create session ids frames
        ip_sessions = log_ip["session_id"]
        ph_sessions = log_ph["session_id"]

        del log_ip, log_ip_tmp, log_ph_tmp

        # Drop unwanted data
        ip_batch = tmp_ip.drop(['session_position', 'session_length',
            'track_id','session_id'], axis=1)

        ph_batch = tmp_ph.drop(['session_position', 'session_length',
            'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
            'no_pause_before_play', 'short_pause_before_play',
            'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
            'hist_user_behavior_n_seekback', 'hist_user_behavior_is_shuffle',
            'hour_of_day', 'date', 'premium', 'context_type',
            'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end',
            'track_id', 'session_id'], axis=1)
        time_list = np.append(time_list,time.process_time())

        del tmp_ip

        # Get one-hot encoding
        track_ip_ext  = pd.concat([ip_batch,self.track_extended], axis = 0, sort=False)
        track_ph_ext  = pd.concat([ph_batch,self.track_extended], axis = 0, sort=False)
        track_ip_ext = pd.get_dummies(track_ip_ext, prefix=['key', 'mode'], columns=['key', 'mode'])
        track_ph_ext = pd.get_dummies(track_ph_ext, prefix=['key', 'mode'], columns=['key', 'mode'])

        n = self.track_extended.shape[0]
        tracks_ip = track_ip_ext[:-n].drop(['key_-1','key_11','mode_minor','session_id'],axis=1)
        tracks_ph = track_ph_ext[:-n].drop(['key_-1','key_11','mode_minor','session_id'],axis=1)

        del ip_batch, ph_batch, track_ip_ext, track_ph_ext

        # Normalize columns
        tmp_ip_normal = (tracks_ip - self.t_min) / (self.t_max - self.t_min)
        tmp_ph_normal = (tracks_ph - self.t_min) / (self.t_max - self.t_min)
        tracks_ip = pd.DataFrame(ip_sessions).join(tmp_ip_normal)
        tracks_ph = pd.DataFrame(ph_sessions).join(tmp_ph_normal)
        time_list = np.append(time_list,time.process_time())

        del tmp_ip_normal, tmp_ph_normal

        # Merge input and prehistory and fill up sessions
        track_ip_grouped = tracks_ip.groupby("session_id")
        track_ph_grouped = tracks_ph.groupby("session_id")
        session_ids = ip_sessions.drop_duplicates().reset_index(drop=True)
        n_rows = len(session_ids) * 20;

        data = np.transpose(np.array([np.zeros(n_rows, dtype=dt) for dt in tracks_ip.dtypes]))
        sessions = pd.DataFrame(data)
        sessions.columns = tracks_ip.columns

        del tracks_ip, data

        for ix, item in session_ids.items():
            ip = track_ip_grouped.get_group(item)
            ph = track_ph_grouped.get_group(item)
            L_ip = len(ip)
            L_ph = len(ph)
            sessions.iloc[ix*20:(ix*20+L_ph), :] = ph.values
            sessions.iloc[(ix*20+L_ph):(ix*20+L_ph+L_ip), :] = ip.values
        time_list = np.append(time_list,time.process_time())

        # Create skip information and output vector
        k_y = tmp_ph[["session_id", "skip_2"]]
        ky_grouped = k_y.groupby("session_id")
        kys = []
        for item in k_y["session_id"].drop_duplicates():
            chk = ky_grouped.get_group(item)
            L_s = len(chk)
            ky_tmp = np.array(chk["skip_2"])*2-1
            ky_tmp = ky_tmp.reshape((-1,1))
            kys.append(np.pad(ky_tmp, [(0,20-L_s),(0,0)], 'constant'))
        kys = np.array(kys)
        kys_shape = kys.shape
        kys = np.reshape(kys,(kys_shape[0]*kys_shape[1],kys_shape[2]))
        df_kys = pd.DataFrame(data=kys, columns=["y1"], dtype="int")
        result = pd.concat([sessions, df_kys], axis=1)
        time_list = np.append(time_list,time.process_time())

        del k_y, ky_grouped, kys, tmp_ph

        #create sessions_based_information
        #Remove unwanted information
        session_fixed = log_ph.drop(['session_position', 'track_id',
           'skip_1', 'skip_2', 'skip_3', 'not_skipped', 'context_switch',
           'no_pause_before_play', 'short_pause_before_play',
           'long_pause_before_play', 'hist_user_behavior_n_seekfwd',
           'hist_user_behavior_n_seekback',
           'hist_user_behavior_reason_start', 'hist_user_behavior_reason_end'], axis=1)
        session_fixed_ids = session_fixed["session_id"].drop_duplicates().reset_index(drop=True)
        session_grouped = session_fixed.groupby("session_id")

        del log_ph

        #Create empty variable to store values in
        n_rows = len(session_fixed_ids);
        data = np.transpose(np.array([np.empty(n_rows, dtype=dt) for dt in session_fixed.dtypes]))
        session_single = pd.DataFrame(data)
        session_single.columns = session_fixed.columns

        del session_fixed, data

        #Fill in session information with information of last element of first half
        for ix, item in session_fixed_ids.items():
            chk = session_grouped.get_group(item)
            session_single.iloc[ix] = chk.iloc[-1]

        #Modify date and perform one-hot-encoding
        session_single['date'] = pd.to_datetime(session_single['date'],
            errors='coerce').apply(lambda x: x.dayofyear)

        session_ext  = pd.concat([session_single,self.session_extended], axis = 0)
        session_ext = pd.get_dummies(session_ext, prefix=['shuffle', 'premium', 'context'],
            columns=['hist_user_behavior_is_shuffle', 'premium','context_type'])

        n = self.session_extended.shape[0]
        session_single = session_ext[:-n].drop(['session_id','shuffle_False',
            'premium_False','context_user_collection'],axis=1)

        del session_ext

        # session_single = pd.get_dummies(session_single,
        #     prefix=['shuffle', 'premium', 'context'],
        #     columns=['hist_user_behavior_is_shuffle', 'premium','context_type'])
        # session_single = session_single.drop(['session_id','shuffle_False',
        #     'premium_False','context_user_collection'],axis=1)

        #Normalize data
        tmp3 = (session_single - self.s_min) / (self.s_max - self.s_min)
        session_finish = pd.DataFrame(session_ids).join(tmp3)

        del tmp3

        # save to csv file
        print("Start saving to file %s" % os.path.basename(file))
        output_file_path = path_output + '/' + os.path.basename(file)
        output_file_path = output_file_path.replace('input_','')
        result.to_csv(output_file_path, index=False)
        time_list = np.append(time_list,time.process_time())

        #Save to csv file
        output_file_path = path_output + '/session_'+ os.path.basename(file)
        output_file_path = output_file_path.replace('input_','')
        session_finish.to_csv(output_file_path, index=False)
        time_list = np.append(time_list,time.process_time())
        print("Finished saving to file %s" % os.path.basename(file))

        # Print times
        time_list = (time_list - start) / 60;
        if self.verbosity:
            # print(result.head())
            print("Loading of file %s finished" % os.path.basename(file))
            print("Time [in minutes] used for loading batch: %4.2f, Merging tracks: %4.2f" % (time_list[0],time_list[1]),
            "Drop unwanted columns: %4.2f, Normalize data: %4.2f," % (time_list[2],time_list[3]),
            "Fill up sessions: %4.2f, Create skip information: %4.2f," % (time_list[4],time_list[5]),
            "Save to file: %4.2f, Create Session fixed information: %4.2f" % (time_list[6],time_list[7]))

        return

    def check_existance(self,path,verbosity=True):
        exists = os.path.isfile(path) or os.path.exists(path)

        if not exists and verbosity:
            print('File not found ' + path)
        return exists
