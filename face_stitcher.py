import cv2
import os
import math
import numpy as np
import pickle
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from face_detector import *
from illumination_enhance import histeq, oneHDR


class FaceStitcher:
    img_count = 0
    ZEROONE = 1
    HISTEQ = 2
    HDR = 3

    def __init__(self, template):
        self.ratio = 0.6
        self.min_match = 10
        self.sift = cv2.SIFT_create()
        self.template = template.copy()
        self.window_height = self.template.shape[0]
        self.window_width = self.template.shape[1]
        self.kp, self.des = self.sift.detectAndCompute(
            FaceStitcher.illu_norm(self.template, mode=FaceStitcher.HISTEQ), None)

    def registration(self, img, plot=False):
        kp1, des1 = self.kp, self.des
        kp2, des2 = self.sift.detectAndCompute(FaceStitcher.illu_norm(img, FaceStitcher.HISTEQ), None)
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        is_good = []
        for i, j in good_points:
            if abs(kp1[j].pt[0] - kp2[i].pt[0]) < self.window_width / 20 and abs(
                    kp1[j].pt[1] - kp2[i].pt[1]) < self.window_height / 20 and (kp1[j].pt[1] < 0.21 * self.window_height or kp1[j].pt[1] > 0.43 * self.window_height) :
                is_good.append(True)
            else:
                is_good.append(False)
        good_points = [good_points[i] for i in range(len(good_points)) if is_good[i] is True]
        good_matches = [good_matches[i] for i in range(len(good_matches)) if is_good[i] is True]
        if plot:
            img3 = cv2.drawMatchesKnn(self.template, kp1, img, kp2, good_matches, None, flags=2)
            cv2.imwrite('./video_clip/matching_{}.jpg'.format(FaceStitcher.img_count), img3)
        FaceStitcher.img_count += 1

        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            a, b = image1_kp.mean(axis=0) - image2_kp.mean(axis=0)
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)
        else:
            H = np.identity(3)
        return H

    def stitching(self, img, plot_kp=False):
        H = self.registration(img, plot=plot_kp)
        height_img1 = self.window_height
        width_img1 = self.window_width
        width_img2 = img.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:self.template.shape[0], 0:self.template.shape[1], :] = self.template
        panorama2 = cv2.warpPerspective(img, H, (width_panorama, height_panorama))
        mask = np.moveaxis(np.tile(panorama2.sum(axis=2) <= 0, (3, 1, 1)), 0, 2)
        result = panorama1 * mask + panorama2
        return result[0:self.window_height, 0:self.window_width, :].astype('uint8')

    @classmethod
    def illu_norm(cls, image, mode=ZEROONE):
        frame_norm = image.astype('float')
        if mode == FaceStitcher.ZEROONE:
            for channel in range(frame_norm.shape[2]):
                area = frame_norm[:, :, channel]
                frame_norm[:, :, channel] = (area - area.min()) / (area.max() - area.min()) * 255
        elif mode == FaceStitcher.HISTEQ:
            for channel in range(frame_norm.shape[2]):
                frame_norm[:, :, channel], _ = histeq(frame_norm[:, :, channel])
        elif mode == FaceStitcher.HDR:
            frame_norm = oneHDR(frame_norm) * 255
        return frame_norm.astype('uint8')


class ExtendFaceStitcher(FaceStitcher):
    def __init__(self, template, landmarks):
        super().__init__(template)
        self.ldmk = landmarks
        self.H = np.identity(3)
        self.error = 0

    def extend_registration(self, img, landmarks, plot=False):
        kp1, des1, ldmk1 = self.kp, self.des, self.ldmk
        kp2, des2 = self.sift.detectAndCompute(super().illu_norm(img, FaceStitcher.HISTEQ), None)
        ldmk2 = landmarks
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            i1, j1 = m1.trainIdx, m1.queryIdx
            x1_distance = abs(kp1[j1].pt[0] - kp2[i1].pt[0])
            y1_distance = abs(kp1[j1].pt[1] - kp2[i1].pt[1])
            total1_distance = math.sqrt((m1.distance / 180) ** 2 + (x1_distance / self.window_width) ** 2 +
                                        (y1_distance / self.window_height) ** 2)
            i2, j2 = m2.trainIdx, m2.queryIdx
            x2_distance = abs(kp1[j2].pt[0] - kp2[i2].pt[0])
            y2_distance = abs(kp1[j2].pt[1] - kp2[i2].pt[1])
            total2_distance = math.sqrt((m2.distance / 180) ** 2 + (x2_distance / self.window_width) ** 2 +
                                        (y2_distance / self.window_height) ** 2)
            if total1_distance < self.ratio * total2_distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        is_good = []
        num_good_points = len(good_points)
        location_distances = np.zeros(num_good_points)
        feature_distances = np.zeros(num_good_points)
        for index in range(num_good_points):
            i, j = good_points[index]
            x_distance = abs(kp1[j].pt[0] - kp2[i].pt[0])
            y_distance = abs(kp1[j].pt[1] - kp2[i].pt[1])
            location_distances[index] = math.sqrt(x_distance ** 2 + y_distance ** 2)
            feature_distances[index] = good_matches[index][0].distance
        location_distances = (location_distances - location_distances.mean()) ** 2 / location_distances.std() ** 2
        feature_distances = (feature_distances - feature_distances.mean()) ** 2 / feature_distances.std() ** 2
        gaussian_distances = 5.0 * location_distances + feature_distances
        sorted_distance = gaussian_distances.copy()
        sorted_distance.sort()
        threshold = sorted_distance[int(num_good_points * 0.5)]
        for index in range(num_good_points):
            is_good.append(gaussian_distances[index] < threshold)
        good_points = [good_points[i] for i in range(len(good_points)) if is_good[i]]
        good_matches = [good_matches[i] for i in range(len(good_matches)) if is_good[i]]
        if plot:
            img3 = cv2.drawMatchesKnn(self.template, kp1, img, kp2, good_matches, None, flags=2)
            cv2.imwrite('./video_clip/matching_{}.jpg'.format(FaceStitcher.img_count), img3)
        FaceStitcher.img_count += 1

        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image1_ldmk = np.float32(swap_x_y(ldmk1))
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        image2_ldmk = np.float32(swap_x_y(ldmk2))
        if len(good_points) > self.min_match:
            A_kp = np.zeros((2 * image2_kp.shape[0], 6))
            b_kp = np.zeros(2 * image2_kp.shape[0])
            for row_idx in range(image2_kp.shape[0]):
                A_kp[2 * row_idx, :2] = image2_kp[row_idx]
                A_kp[2 * row_idx, 2] = 1
                A_kp[2 * row_idx + 1, 3:5] = image2_kp[row_idx]
                A_kp[2 * row_idx + 1, 5] = 1
                b_kp[2 * row_idx] = image1_kp[row_idx, 0]
                b_kp[2 * row_idx + 1] = image1_kp[row_idx, 1]
            t = np.linalg.inv(A_kp.T.dot(A_kp)).dot(A_kp.T).dot(b_kp)
            H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])
        else:
            A_kp = np.zeros((2 * image2_ldmk.shape[0], 6))
            b_kp = np.zeros(2 * image2_ldmk.shape[0])
            for row_idx in range(image2_ldmk.shape[0]):
                A_kp[2 * row_idx, :2] = image2_ldmk[row_idx]
                A_kp[2 * row_idx, 2] = 1
                A_kp[2 * row_idx + 1, 3:5] = image2_ldmk[row_idx]
                A_kp[2 * row_idx + 1, 5] = 1
                b_kp[2 * row_idx] = image1_ldmk[row_idx, 0]
                b_kp[2 * row_idx + 1] = image1_ldmk[row_idx, 1]
            t = np.linalg.inv(A_kp.T.dot(A_kp)).dot(A_kp.T).dot(b_kp)
            H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])
        return H

    def extend_stitching(self, img, landmarks, plot_kp=False):
        H = self.extend_registration(img, landmarks, plot=plot_kp)
        height_img1 = self.window_height
        width_img1 = self.window_width
        width_img2 = img.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:self.template.shape[0], 0:self.template.shape[1], :] = self.template
        panorama2 = cv2.warpPerspective(img, H, (width_panorama, height_panorama))
        mask = np.moveaxis(np.tile(panorama2.sum(axis=2) <= 0, (3, 1, 1)), 0, 2)
        result = panorama1 * mask + panorama2
        return result[0:self.window_height, 0:self.window_width, :].astype('uint8')

    def apply_warp(self, points, Y_FIRST=True):
        if Y_FIRST:
            anchor = swap_x_y(points)
            coordinate_matrix = np.ones((anchor.shape[1] + 1, anchor.shape[0]))
            coordinate_matrix[:2, :] = anchor.T
            aligned_matrix = self.H.dot(coordinate_matrix)
            return swap_x_y(aligned_matrix[:2, :].T)
        else:
            anchor = points.copy()
            coordinate_matrix = np.ones((anchor.shape[1] + 1, anchor.shape[0]))
            coordinate_matrix[:2, :] = points.T
            aligned_matrix = self.H.dot(coordinate_matrix)
            return aligned_matrix[:2, :].T

    def transform_error(self, template_kp, query_kp, Y_FIRST=True):
        aligned_kp = self.apply_warp(query_kp, Y_FIRST=Y_FIRST)
        return np.linalg.norm(template_kp - aligned_kp) / template_kp.shape[0]

    def update_landmarks(self, ldmk, Y_FIRST=True):
        aligned_ldmk = self.apply_warp(ldmk, Y_FIRST=Y_FIRST)
        if Y_FIRST:
            self.ldmk = aligned_ldmk
        else:
            self.ldmk = swap_x_y(aligned_ldmk)
        self.error = 0
        return aligned_ldmk


class CascadeFaceStitcher(FaceStitcher):
    def __init__(self, template, landmarks):
        super().__init__(template)
        self.ldmk = np.float32(swap_x_y(landmarks))
        self.threshold = 1.0

        self.kps = [self.kp]
        self.deses = [self.des]
        self.ldmks = [self.ldmk]
        self.Hs = [np.identity(3)]
        self.errors = [0.0]

    def cal_homography(self, kp1, des1, ldmk1, kp2, des2, ldmk2):
        if kp1 == kp2 and des1 == des2:
            return np.identity(3), kp1, kp2

        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good_points = []
        good_matches = []
        for m1, m2 in raw_matches:
            i1, j1 = m1.trainIdx, m1.queryIdx
            x1_distance = abs(kp1[j1].pt[0] - kp2[i1].pt[0])
            y1_distance = abs(kp1[j1].pt[1] - kp2[i1].pt[1])
            total1_distance = math.sqrt((m1.distance / 180) ** 2 + (x1_distance / self.window_width) ** 2 +
                                        (y1_distance / self.window_height) ** 2)
            i2, j2 = m2.trainIdx, m2.queryIdx
            x2_distance = abs(kp1[j2].pt[0] - kp2[i2].pt[0])
            y2_distance = abs(kp1[j2].pt[1] - kp2[i2].pt[1])
            total2_distance = math.sqrt((m2.distance / 180) ** 2 + (x2_distance / self.window_width) ** 2 +
                                        (y2_distance / self.window_height) ** 2)
            if total1_distance < self.ratio * total2_distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])

        is_good = []
        num_good_points = len(good_points)
        location_distances = np.zeros(num_good_points)
        feature_distances = np.zeros(num_good_points)
        for index in range(num_good_points):
            i, j = good_points[index]
            x_distance = abs(kp1[j].pt[0] - kp2[i].pt[0])
            y_distance = abs(kp1[j].pt[1] - kp2[i].pt[1])
            location_distances[index] = math.sqrt(x_distance ** 2 + y_distance ** 2)
            feature_distances[index] = good_matches[index][0].distance
        location_distances = (location_distances - location_distances.mean()) ** 2 / location_distances.std() ** 2
        feature_distances = (feature_distances - feature_distances.mean()) ** 2 / feature_distances.std() ** 2
        gaussian_distances = location_distances * 3.0 + feature_distances
        sorted_distance = gaussian_distances.copy()
        sorted_distance.sort()
        threshold = sorted_distance[int(num_good_points * 0.6)]
        for index in range(num_good_points):
            is_good.append(gaussian_distances[index] < threshold)
        good_points = [good_points[i] for i in range(len(good_points)) if is_good[i]]
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
        if len(good_points) > self.min_match:
            # affine transform
            A_kp = np.zeros((2 * image2_kp.shape[0], 6))
            b_kp = np.zeros(2 * image2_kp.shape[0])
            for row_idx in range(image2_kp.shape[0]):
                A_kp[2 * row_idx, :2] = image2_kp[row_idx]
                A_kp[2 * row_idx, 2] = 1
                A_kp[2 * row_idx + 1, 3:5] = image2_kp[row_idx]
                A_kp[2 * row_idx + 1, 5] = 1
                b_kp[2 * row_idx] = image1_kp[row_idx, 0]
                b_kp[2 * row_idx + 1] = image1_kp[row_idx, 1]
            t = np.linalg.inv(A_kp.T.dot(A_kp)).dot(A_kp.T).dot(b_kp)
            H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])
            # translation transform
            # a, b = image1_kp.mean(axis=0) - image2_kp.mean(axis=0)
            # H = np.array([[1, 0, a], [0, 1, b], [0, 0, 1]])
            # homography transform
            # H, status = cv2.findHomography(image2_kp, image1_kp, cv2.LMEDS, 5.0)
        else:
            A_kp = np.zeros((2 * ldmk2.shape[0], 6))
            b_kp = np.zeros(2 * ldmk2.shape[0])
            for row_idx in range(ldmk2.shape[0]):
                A_kp[2 * row_idx, :2] = ldmk2[row_idx]
                A_kp[2 * row_idx, 2] = 1
                A_kp[2 * row_idx + 1, 3:5] = ldmk2[row_idx]
                A_kp[2 * row_idx + 1, 5] = 1
                b_kp[2 * row_idx] = ldmk1[row_idx, 0]
                b_kp[2 * row_idx + 1] = ldmk1[row_idx, 1]
            t = np.linalg.inv(A_kp.T.dot(A_kp)).dot(A_kp.T).dot(b_kp)
            H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])
            # a, b = ldmk1.mean(axis=0) - ldmk2.mean(axis=0)
            # H = np.array([[1, 0, a], [0, 1, b], [0, 0, 1]])
            # H, status = cv2.findHomography(ldmk2, ldmk1, cv2.LMEDS, 5.0)
        return H, image1_kp, image2_kp

    def apply_warp(self, H, points, Y_FIRST=True):
        if Y_FIRST:
            anchor = swap_x_y(points)
            coordinate_matrix = np.ones((anchor.shape[1] + 1, anchor.shape[0]))
            coordinate_matrix[:2, :] = anchor.T
            aligned_matrix = H.dot(coordinate_matrix)
            return swap_x_y(aligned_matrix[:2, :].T)
        else:
            anchor = points.copy()
            coordinate_matrix = np.ones((anchor.shape[1] + 1, anchor.shape[0]))
            coordinate_matrix[:2, :] = points.T
            aligned_matrix = H.dot(coordinate_matrix)
            return aligned_matrix[:2, :].T

    def transform_error(self, template_kp, H, query_kp, Y_FIRST=True):
        aligned_kp = self.apply_warp(H, query_kp, Y_FIRST=Y_FIRST)
        return np.linalg.norm(template_kp - aligned_kp) / template_kp.shape[0]

    def cascade_registration(self, img, landmarks, plot=False):
        frame_id = len(self.Hs)
        kp1, des1, ldmk1 = self.kp, self.des, self.ldmk
        kp2, des2, ldmk2 = self.kps[frame_id // 2], self.deses[frame_id // 2], self.ldmks[frame_id // 2]
        kp3, des3 = self.sift.detectAndCompute(super().illu_norm(img, FaceStitcher.HISTEQ), None)
        ldmk3 = np.float32(swap_x_y(landmarks))
        self.kps.append(kp3)
        self.deses.append(des3)
        self.ldmks.append(ldmk3)

        FaceStitcher.img_count += 1
        if frame_id == 1:
            H_direct, kp1_p, kp3_p = self.cal_homography(kp1, des1, ldmk1, kp3, des3, ldmk3)
            self.Hs.append(H_direct)
            self.errors.append(self.transform_error(kp1_p, H_direct, kp3_p))
            return H_direct
        else:
            H_direct, kp1_p, kp3_p = self.cal_homography(kp1, des1, ldmk1, kp3, des3, ldmk3)
            error_kp_direct = self.transform_error(kp1_p, H_direct, kp3_p)
            error_ldmk_direct = self.transform_error(ldmk1, H_direct, ldmk3)
            error_direct = error_kp_direct + self.threshold * error_ldmk_direct
            H1 = self.Hs[frame_id // 2]
            H2, kp2_p, kp3_p = self.cal_homography(kp2, des2, ldmk2, kp3, des3, ldmk3)
            H_indirect = H1.dot(H2)
            error_kp_indirect = self.errors[frame_id // 2] + self.transform_error(kp2_p, H2, kp3_p)
            error_ldmk_indirect = self.transform_error(ldmk1, H_indirect, ldmk3)
            error_indirect = error_kp_indirect + self.threshold * error_ldmk_indirect
            if error_direct < error_indirect:
                self.Hs.append(H_direct)
                self.errors.append(error_kp_direct)
                return H_direct
            else:
                self.Hs.append(H_indirect)
                self.errors.append(error_kp_indirect)
                return H_indirect

    def cascade_stitching(self, img, landmarks, plot_kp=False):
        H = self.cascade_registration(img, landmarks, plot=plot_kp)
        height_img1 = self.window_height
        width_img1 = self.window_width
        width_img2 = img.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:self.template.shape[0], 0:self.template.shape[1], :] = self.template
        panorama2 = cv2.warpPerspective(img, H, (width_panorama, height_panorama))
        mask = np.moveaxis(np.tile(panorama2.sum(axis=2) <= 0, (3, 1, 1)), 0, 2)
        result = panorama1 * mask + panorama2
        return result[0:self.window_height, 0:self.window_width, :].astype('uint8')

    def cascade4_registration(self, img, landmarks, plot=False, max_depth=999):
        frame_id = len(self.Hs)
        if frame_id < 2 ** max_depth:
            return self.cascade_registration(img, landmarks, plot=plot)
        kp1, des1, ldmk1 = self.kp, self.des, self.ldmk
        kp3, des3 = self.sift.detectAndCompute(super().illu_norm(img, FaceStitcher.HISTEQ), None)
        ldmk3 = np.float32(swap_x_y(landmarks))
        self.kps.append(kp3)
        self.deses.append(des3)
        self.ldmks.append(ldmk3)

        H_direct, kp1_p, kp3_p = self.cal_homography(kp1, des1, ldmk1, kp3, des3, ldmk3)
        error_kp_direct = self.transform_error(kp1_p, H_direct, kp3_p)
        error_ldmk_direct = self.transform_error(ldmk1, H_direct, ldmk3)
        error_direct = error_kp_direct + self.threshold * error_ldmk_direct

        current_depth = 1
        current_H = H_direct
        current_kp_error = error_kp_direct
        current_total_error = error_direct
        while current_depth <= max_depth and frame_id // 2 ** current_depth != 0:
            kp2, des2, ldmk2 = self.kps[frame_id - frame_id // 2 ** current_depth], \
                               self.deses[frame_id - frame_id // 2 ** current_depth], \
                               self.ldmks[frame_id - frame_id // 2 ** current_depth]

            H1 = self.Hs[frame_id - frame_id // 2 ** current_depth]
            H2, kp2_p, kp3_p = self.cal_homography(kp2, des2, ldmk2, kp3, des3, ldmk3)
            H_indirect = H1.dot(H2)
            error_kp_indirect = self.errors[frame_id // 2] + self.transform_error(kp2_p, H2, kp3_p)
            error_ldmk_indirect = self.transform_error(ldmk1, H_indirect, ldmk3)
            error_indirect = error_kp_indirect + self.threshold * error_ldmk_indirect

            if error_indirect < current_total_error:
                current_H = H_indirect
                current_total_error = error_indirect
                current_kp_error = error_kp_indirect
                current_depth += 1
            else:
                break
        self.Hs.append(current_H)
        self.errors.append(current_kp_error)
        FaceStitcher.img_count += 1
        return current_H

    def cascade4_stitching(self, img, landmarks, plot_kp=False):
        H = self.cascade4_registration(img, landmarks, plot=plot_kp)
        # H = self.cascade4_registration(img, landmarks, plot=plot_kp, max_depth=4)
        height_img1 = self.window_height
        width_img1 = self.window_width
        width_img2 = img.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 + width_img2

        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        panorama1[0:self.template.shape[0], 0:self.template.shape[1], :] = self.template
        panorama2 = cv2.warpPerspective(img, H, (width_panorama, height_panorama))
        mask = np.moveaxis(np.tile(panorama2.sum(axis=2) <= 0, (3, 1, 1)), 0, 2)
        result = panorama1 * mask + panorama2

        return result[0:self.window_height, 0:self.window_width, :].astype('uint8')

def error(distance, ratio):
    x = distance * ratio
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def swap_x_y(matrix):
    template = np.zeros(matrix.shape)
    template[:, 0] = matrix[:, 1]
    template[:, 1] = matrix[:, 0]
    return template

