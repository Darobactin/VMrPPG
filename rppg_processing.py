import os
import sys
import numpy as np
import pickle
from scipy import fftpack, interpolate
from sklearn.decomposition import PCA
from tqdm import tqdm

if __name__ == '__main__':
    database = '3DMAD'
    method_name = '3DMAD_HI_S'
    video_type = 'real'
    feature_type = 'bp'
    video_length = 10  # in seconds
    segment_length = 1  # in seconds
    overlap_step = 0.1  # in seconds
    FPS_norm = 30  # normalized FPS
    extend_scale = int((video_length - segment_length) / overlap_step + 1)
    num_rows = 4
    num_cols = 6
    if len(sys.argv) > 3:
        video_type = sys.argv[1]
        num_rows = int(sys.argv[2])
        num_cols = int(sys.argv[3])
    with open("./models/{}/{}_{}_{}_{}_ST_{}.pkl".format(
            method_name, database, video_type, num_rows, num_cols, feature_type), "rb") as f:
        st_maps_lists = pickle.load(f)
    if not os.path.isdir('/home/jason/data/models/{}'.format(method_name)):
        os.mkdir('/home/jason/data/models/{}'.format(method_name))
    session_id = 0
    for st_maps_list in st_maps_lists:
        session_id += 1
        num_videos = len(st_maps_list)
        # FFT_maps = np.zeros((extend_scale * num_videos, st_maps_list[0].shape[0], FPS_norm * segment_length // 2,
        #                      st_maps_list[0].shape[2] * 2))
        # toeplitz_maps = np.zeros((extend_scale * num_videos, FPS_norm * segment_length // 2,
        #                           FPS_norm * segment_length // 2, st_maps_list[0].shape[2] * 4))
        st_maps = np.zeros((extend_scale * num_videos, st_maps_list[0].shape[0], FPS_norm * segment_length,
                            st_maps_list[0].shape[2] * 2))
        si_maps = np.zeros((extend_scale * num_videos, (num_cols - 1) * num_rows + (num_rows - 1) * num_cols,
                            FPS_norm * segment_length, st_maps_list[0].shape[2] * 2))

        print("session {} video count: {}".format(session_id, num_videos))
        for video_id in tqdm(range(num_videos)):
            num_windows = st_maps_list[video_id].shape[0]
            num_frames = st_maps_list[video_id].shape[1]
            num_channels = st_maps_list[video_id].shape[2]
            FPS = num_frames / video_length
            for window_id in range(num_windows):
                for channel_id in range(num_channels):
                    pulse = st_maps_list[video_id][window_id, :, channel_id]
                    for segment_id in range(extend_scale):
                        pulse_segment = pulse[int(segment_id * overlap_step * FPS):int(
                            segment_id * overlap_step * FPS + segment_length * FPS)]
                        # generate time based encoding
                        x = np.array(range(pulse_segment.shape[0]))
                        f = interpolate.interp1d(x, pulse_segment, kind='quadratic')
                        xnew = np.linspace(0, pulse_segment.shape[0] - 1, FPS_norm * segment_length)
                        pulse_segment_interp = f(xnew)
                        # pulse_segment_fstGradient = np.gradient(pulse_segment_interp, edge_order=2)
                        # pulse_segment_sndGradient = np.gradient(pulse_segment_fstGradient, edge_order=2)
                        pulse_segment_interp_norm = (pulse_segment_interp - pulse_segment_interp.min()) / (
                                pulse_segment_interp.max() - pulse_segment_interp.min())
                        '''# toeplitz encoding
                        if window_id == 13:
                            for i in range(toeplitz_maps.shape[1]):
                                toeplitz_maps[segment_id + video_id * extend_scale, i, :,
                                2 * channel_id] = pulse_segment_interp[i:i + toeplitz_maps.shape[2]]
                                toeplitz_maps[segment_id + video_id * extend_scale, i, :,
                                2 * (channel_id + num_channels)] = pulse_segment_interp_norm[i:i + toeplitz_maps.shape[2]]
                        if window_id == 16:
                            for i in range(toeplitz_maps.shape[1]):
                                toeplitz_maps[segment_id + video_id * extend_scale, i, :,
                                2 * channel_id + 1] = pulse_segment_interp[i:i + toeplitz_maps.shape[2]]
                                toeplitz_maps[segment_id + video_id * extend_scale, i, :,
                                2 * (channel_id + num_channels) + 1] = pulse_segment_interp_norm[i:i + toeplitz_maps.shape[2]]'''
                        # st encoding
                        st_maps[segment_id + video_id * extend_scale, window_id, :, channel_id] = pulse_segment_interp
                        st_maps[segment_id + video_id * extend_scale, window_id, :,
                                channel_id + num_channels] = pulse_segment_interp_norm
                        '''# generate spectral encoding
                        fft_result = np.abs(fftpack.fft(pulse_segment_interp)) / pulse_segment_interp.shape[0]
                        pulse_segment_fft = fft_result[:pulse_segment_interp.shape[0] // 2]
                        pulse_segment_fft_norm = (pulse_segment_fft - pulse_segment_fft.min()) / (
                                pulse_segment_fft.max() - pulse_segment_fft.min())
                        FFT_maps[segment_id + video_id * extend_scale, window_id, :, channel_id] = pulse_segment_fft
                        FFT_maps[segment_id + video_id * extend_scale, window_id, :,
                                 channel_id + num_channels] = pulse_segment_fft_norm
        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_FFT_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(FFT_maps, f, protocol=4)
        print(FFT_maps.shape)

        FFT_maps_abs = FFT_maps[:, :, :, :FFT_maps.shape[3] // 2]
        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_FFTr_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(FFT_maps_abs, f, protocol=4)
        print(FFT_maps_abs.shape)'''

        '''with open("{}_{}_TPL.pkl".format(database, video_type), "wb") as f:
            pickle.dump(toeplitz_maps, f, protocol=4)
        print(toeplitz_maps.shape)'''

        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_STN_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(st_maps, f, protocol=4)
        print(st_maps.shape)

        st_maps_abs = st_maps[:, :, :, :st_maps.shape[3] // 2]
        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_STNr_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(st_maps_abs, f, protocol=4)
        print(st_maps_abs.shape)

        # local (phase) similarity encoding
        '''neighbors = []
        for index in range(num_cols * num_rows):
            if (index + 1) % num_cols != 0:
                neighbors.append((index, index + 1))
            if index // num_rows < (num_rows - 1):
                neighbors.append((index, index + num_cols))
        window_id = 0
        for videoseg_id in tqdm(range(si_maps.shape[0])):
            for channel_id in range(si_maps.shape[3]):
                for rawwindow_id1, rawwindow_id2 in neighbors:
                    pca = PCA(n_components=1)
                    si_maps[videoseg_id, window_id, :, channel_id] = pca.fit_transform(
                        np.array([st_maps[videoseg_id, rawwindow_id1, :, channel_id],
                                  st_maps[videoseg_id, rawwindow_id2, :, channel_id]]).T)[:, 0]
        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_SIP_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(si_maps, f, protocol=4)
        print(si_maps.shape)

        si_maps_abs = si_maps[:, :, :, :si_maps.shape[3] // 2]
        with open("/home/jason/data/models/{}/{}_{}_{}_{}_{}_SIPr_{}.pkl".format(
                method_name, database, video_type, num_rows, num_cols, session_id, feature_type), "wb") as f:
            pickle.dump(si_maps_abs, f, protocol=4)
        print(si_maps_abs.shape)'''
