import os
import pickle
import sys
import time
import warnings
import multiprocessing

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack, signal
from face_detector import SeetaFace2, SeetaFace6
from face_stitcher import FaceStitcher, ExtendFaceStitcher, CascadeFaceStitcher


SAVE_SEGMENT_PREVIEW = False
TEST_MODE = 'Normal'
PROCESS_MODE = None

num_rows = 4
num_cols = 6
Num_rows = num_rows
Num_cols = num_cols
topbar = 0.05
bottombar = 0.1
leftbar = -0.05
rightbar = -0.05

modelPath1 = '/home/jason/databases/seetaface2_models/fd_2_00.dat'
modelPath2 = '/home/jason/databases/seetaface2_models/pd_2_00_pts81.dat'
seetaface_type = 6


class AlignQuality:
    def __init__(self):
        self.images = np.array([])
        self.frame_count = 0
        self.std = 0

    def init(self, img_height, img_width, img_frames):
        self.images = np.zeros((img_height, img_width, img_frames), dtype=np.float)

    def append(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.images[:, :, self.frame_count] = img_gray
        self.frame_count += 1

    def analyze(self):
        std_map = self.images.std(axis=2)
        return std_map.mean()

def detrend_filtering(pulse):
    pulse_detrend = signal.detrend(pulse)
    return pulse_detrend

def lowpass_filtering(pulse, high, FPS, N=8):
    pulse_detrend = detrend_filtering(pulse)
    Wn_low = 2 * high / FPS
    b, a = signal.butter(N, Wn_low, 'lowpass')
    pulse_lowpass = signal.filtfilt(b, a, pulse_detrend)
    return pulse_lowpass

def bandpass_filtering(pulse, low, high, FPS, N=8):
    pulse_lowpass = lowpass_filtering(pulse, high, FPS, N)
    Wn_high = 2 * low / FPS
    b, a = signal.butter(N, Wn_high, 'highpass')
    pulse_highpass = signal.filtfilt(b, a, pulse_lowpass)
    return pulse_highpass

def extract_seetaface(queue, st_list, video_path, lock):
    queue.put("Busy")
    video_name = video_path[-12:-4]
    camera = cv2.VideoCapture(video_path)
    num_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
    if num_frames <= 1:
        return
    FPS = int(camera.get(cv2.CAP_PROP_FPS))
    norm_mean_info = []
    norm_std_info = []
    face = None
    face_previous = None
    stitcher = None
    align_quality = AlignQuality()
    if seetaface_type == 2:
        seeta = SeetaFace2()
        seeta.landmark_init([modelPath1, modelPath2])
    else:
        seeta = SeetaFace6()
        seeta.landmark_init()
    mean_values = []  # [regions, frames, channels]
    for _ in range((num_rows + 1) * num_cols):
        mean_values.append([])

    (grabbed, frame) = camera.read()
    frame_count = 0
    while grabbed:
        direction = video_name[:2]
        if direction == "04":
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
        elif direction == "05" or direction == "06":
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 0)

        if PROCESS_MODE == "FRAMENORM":
            frame_YUV = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            frame_LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            norm_mean_info.append(
                np.array([frame_YUV[:, :, 0].mean(), frame[:, :, 1].mean(), frame_LAB[:, :, 0].mean()]))
            norm_std_info.append(np.array([frame_YUV[:, :, 0].std(), frame[:, :, 1].std(), frame_LAB[:, :, 0].std()]))

        startX, startY, endX, endY = seeta.detect(frame)
        if startX >= 0 and endX >= 0:
            face = frame[startY:endY, startX:endX, :]
            face_previous = face.copy()
        else:
            face = face_previous
        if seetaface_type == 2:
            ldmk = seeta.landmark(face, boost=True)
        else:
            ldmk = seeta.landmark(frame)
            ldmk[:, 1] -= startX
            ldmk[:, 0] -= startY
        if frame_count == 0:
            align_quality.init(face.shape[0], face.shape[1], num_frames)
            stitcher = CascadeFaceStitcher(face, ldmk)
            ROI_startX = ldmk[:, 1].min()
            ROI_startY = ldmk[:, 0].min()
            ROI_endX = ldmk[:, 1].max()
            ROI_endY = ldmk[:, 0].max()
        else:
            face = stitcher.cascade4_stitching(face, ldmk, plot_kp=False)
        align_quality.append(face)
        if TEST_MODE is not None:
            cv2.imwrite("./video_clip/stitched_{}.png".format(frame_count), face)

        cols = np.array([ROI_startX + leftbar * (ROI_endX - ROI_startX) + i / num_cols * (ROI_endX - ROI_startX) * (
                1 - leftbar - rightbar) for i in range(num_cols + 1)]).astype('int')
        rows = np.array([ROI_startY + topbar * (ROI_endY - ROI_startY) + i / num_rows * (ROI_endY - ROI_startY) * (
                1 - topbar - bottombar) for i in range(-1, num_rows + 1)]).astype('int')
        cols[0] = max(cols[0], 0)
        cols[-1] = min(cols[-1], face.shape[1])
        rows[0] = max(rows[0], 0)
        rows[-1] = min(rows[-1], face.shape[0])
        for i in range(num_rows + 1):
            for j in range(num_cols):
                ROI = face[rows[i]:rows[i + 1], cols[j]:cols[j + 1], :]
                ROI_YUV = cv2.cvtColor(ROI, cv2.COLOR_BGR2YUV)
                ROI_RGB = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
                ROI_LAB = cv2.cvtColor(ROI, cv2.COLOR_BGR2LAB)
                ROI_all = np.concatenate((ROI_YUV, ROI_RGB, ROI_LAB), axis=2)
                if TEST_MODE in ["RGB", "Normal"]:
                    mean_values[num_cols * i + j].append(ROI_RGB.mean(axis=(0, 1)))
                elif TEST_MODE in ["LAB"]:
                    mean_values[num_cols * i + j].append(ROI_LAB.mean(axis=(0, 1)))
                else:
                    mean_values[num_cols * i + j].append(ROI_all.mean(axis=(0, 1)))

                if SAVE_SEGMENT_PREVIEW:
                    cv2.imwrite("./video_clip/{}_{}.png".format(i, j), ROI)

        if SAVE_SEGMENT_PREVIEW:
            im = face.copy()
            for col in cols:
                cv2.line(im, (col, rows[0]), (col, rows[-1]), (255, 255, 0), 2)
            for row in rows:
                cv2.line(im, (cols[0], row), (cols[-1], row), (255, 255, 0), 2)
            cv2.imwrite("./video_clip/seg.png", im)
            os._exit(0)

        frame_count += 1
        (grabbed, frame) = camera.read()

        if TEST_MODE is not None:
            if frame_count >= 300:
                break
    for i in range(len(mean_values)):
        mean_values[i] = np.array(mean_values[i])
    mean_values = np.array(mean_values)
    mean_values_bp = np.zeros(mean_values.shape)
    mean_values_FFT = np.zeros((mean_values.shape[0], mean_values.shape[1] // 5, mean_values.shape[2]))
    
    for i in range(mean_values.shape[0]):
        for j in range(mean_values[i].shape[1]):
            mean_values_bp[i, :, j] = bandpass_filtering(mean_values[i, :, j], 0.85, 3.5, FPS)
            mean_values_FFT[i, :, j] = np.abs(fftpack.fft(mean_values_bp[i, :, j]) / mean_values_bp.shape[1])[:mean_values_bp.shape[1] // 5]

    lock.acquire()
    st_list.append([video_name, mean_values, mean_values_bp, mean_values_FFT])
    lock.release()

    if TEST_MODE in ["MSR", "Normal"]:
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 0], 'r')
        plt.title('Left Red')
        plt.savefig("./video_clip/{}_Left_Red.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[13].shape[0])], mean_values_FFT[13, :, 0], color='r', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Left_Red.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 1], 'g')
        plt.title('Left Green')
        plt.savefig("./video_clip/{}_Left_Green.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[13].shape[0])], mean_values_FFT[13, :, 1], color='g', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Left_Green.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 2], 'b')
        plt.title('Left Blue')
        plt.savefig("./video_clip/{}_Left_Blue.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[13].shape[0])], mean_values_FFT[13, :, 2], color='b', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Left_Blue.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 0], 'r')
        plt.title('Right Red')
        plt.savefig("./video_clip/{}_Right_Red.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[16].shape[0])], mean_values_FFT[16, :, 0], color='r', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Right_Red.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 1], 'g')
        plt.title('Right Green')
        plt.savefig("./video_clip/{}_Right_Green.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[16].shape[0])], mean_values_FFT[16, :, 1], color='g', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Right_Green.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 2], 'b')
        plt.title('Right Blue')
        plt.savefig("./video_clip/{}_Right_Blue.png".format(TEST_MODE))
        plt.close()
        plt.plot([i / 10 for i in range(mean_values_FFT[16].shape[0])], mean_values_FFT[16, :, 2], color='b', linewidth=3)
        plt.ylim(0, 0.08)
        plt.savefig("./video_clip/{}_FFT_Right_Blue.png".format(TEST_MODE))
        plt.close()
    elif TEST_MODE in ["LAB", "LAB_INTENSITY"]:
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 0], 'r')
        plt.ylim(-1, 1.25)
        plt.title('Left L')
        plt.savefig("./video_clip/{}_Left_L.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 1], 'g')
        plt.ylim(-1, 1.25)
        plt.title('Left A')
        plt.savefig("./video_clip/{}_Left_A.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[13].shape[0]), mean_values[13, :, 2], 'b')
        plt.ylim(-1, 1.25)
        plt.title('Left B')
        plt.savefig("./video_clip/{}_Left_B.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 0], 'r')
        plt.title('Right L')
        plt.savefig("./video_clip/{}_Right_L.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 1], 'g')
        plt.title('Right A')
        plt.savefig("./video_clip/{}_Right_A.png".format(TEST_MODE))
        plt.close()
        plt.plot(range(mean_values[16].shape[0]), mean_values[16, :, 2], 'b')
        plt.title('Right B')
        plt.savefig("./video_clip/{}_Right_B.png".format(TEST_MODE))
        plt.close()
        print(mean_values[13, :, 0].std(), mean_values[13, :, 1].std(), mean_values[13, :, 2].std())
        print(mean_values[16, :, 0].std(), mean_values[16, :, 1].std(), mean_values[16, :, 2].std())

    camera.release()
    print("{} {} Frame Count: {} FPS: {} Shape: {} std: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                           video_name, num_frames, FPS, mean_values.shape, align_quality.analyze()))
    
    queue.get()

if __name__ == '__main__':
    num_rows -= 1  # (num_rows - 1) rows inside the face landmark region
    database = 'database_name'
    video_type = 'real or attack'
    videos_dir = "/home/jason/databases/database_name/{}".format(video_type)
    videos_path = []
    for parent, dirnames, filenames in os.walk(videos_dir):
        for filename in filenames:
            if filename[-4:] == ".avi" or filename[-4:] == ".mov" or filename[-4:] == ".mp4":
                videos_path.append(os.path.join(parent, filename))
    process_list = []
    ST_maps_list = multiprocessing.Manager().list()
    lock = multiprocessing.Manager().Lock()
    queue = multiprocessing.Queue(16)
    for num in range(len(videos_path)):
        process = multiprocessing.Process(target=extract_seetaface, args=(queue, ST_maps_list, videos_path[num], lock))
        process_list.append(process)
    job_count = 0
    for process in process_list:
        process.start()
        job_count += 1
    for process in process_list:
        process.join()
    with open("./models/{}_{}_{}_{}_ST.pkl".format(database, video_type, Num_rows, Num_cols), "wb") as f:
        pickle.dump(list(ST_maps_list), f)
    print("Finished. Length:{}".format(len(ST_maps_list)))
