import sys
import pickle
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    database = 'HKBUv2'
    video_types = ['real', 'attack']  # real or attack
    num_rows = 4
    num_cols = 6
    for video_type in video_types:
        with open("../models/{}_{}_{}_{}_ST.pkl".format(database, video_type, num_rows, num_cols), "rb") as f:
            st_maps_list = pickle.load(f)
        num_videos = len(st_maps_list)
        print(num_videos)
        st_maps_list.sort(key=lambda x: x[0])
        with open("../models/{}_{}_{}_{}_ST.txt".format(database, video_type, num_rows, num_cols), "w") as f:
            for st_map_list in st_maps_list:
                f.write("{}\n".format(st_map_list[0]))

        st_session_maps_orig = []
        st_session_maps_bp = []
        st_session_maps_fft = []
        for i in range(12):  # 17 for 3DMAD, 12 for HKBUv2
            st_session_maps_orig.append([])
            st_session_maps_bp.append([])
            st_session_maps_fft.append([])

        for i in tqdm(range(num_videos)):
            session_id = int(st_maps_list[i][0][6:8]) - 1  # :3 for 3DMAD, 6:8 for HKBUv2
            st_session_maps_orig[session_id].append(st_maps_list[i][1])
            st_session_maps_bp[session_id].append(st_maps_list[i][2])
            st_session_maps_fft[session_id].append(st_maps_list[i][3])

        print([len(i) for i in st_session_maps_orig])

        with open("../models/{}_{}_{}_{}_ST_orig.pkl".format(database, video_type, num_rows, num_cols), "wb") as f:
            pickle.dump(st_session_maps_orig, f)
        with open("../models/{}_{}_{}_{}_ST_bp.pkl".format(database, video_type, num_rows, num_cols), "wb") as f:
            pickle.dump(st_session_maps_bp, f)
        with open("../models/{}_{}_{}_{}_ST_fft.pkl".format(database, video_type, num_rows, num_cols), "wb") as f:
            pickle.dump(st_session_maps_fft, f)
