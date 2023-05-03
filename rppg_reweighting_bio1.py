import numpy as np
import pickle
import sys
import os

database = '3DMAD'
video_types = ['attack', 'real']  # real or attack
method_name = "3DMAD_OI_S_ALL1S_EQ"
method_processed = method_name[:-2] + "BIO1"
feature_type = 'bp'
model_dir = "./models/"
train_mode = "STN"
num_subjects = 17
num_rows = 4
num_cols = 6

for video_type in video_types:
    with open("bio_weights_1.pkl", "rb") as f:
        weights = pickle.load(f)
    print(weights.shape)

    for subject_id in range(1, num_subjects + 1):
        if database == 'HKBUv1p' and video_type == 'real' and subject_id == 8:
            continue
        with open("{}{}/{}_{}_{}_{}_{}_{}_{}.pkl".format(
                            model_dir, method_name, database, video_type, num_rows, num_cols, subject_id, train_mode, feature_type), "rb") as f:
            st_maps = pickle.load(f)
        print(st_maps.shape)

        for window_id in range(weights.shape[0]):
            st_maps[:, window_id, :, :] *= weights[window_id]

        if not os.path.isdir("{}{}".format(model_dir, method_processed)):
            os.mkdir("{}{}".format(model_dir, method_processed))
        with open("{}{}/{}_{}_{}_{}_{}_{}_{}.bio.pkl".format(
                            model_dir, method_processed, database, video_type, num_rows, num_cols, subject_id, train_mode, feature_type), "wb") as f:
            pickle.dump(st_maps, f, protocol=4)
        print(st_maps.shape)
