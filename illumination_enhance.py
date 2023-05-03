from datetime import datetime
import numpy as np
import cv2
import warnings
from math import ceil
from scipy.sparse import spdiags
from PIL import Image
from scipy.optimize import fminbound
from scipy.stats import entropy
from scipy.sparse.linalg import spsolve

filepath = 'dim1.png'

def computeTextureWeights(fin, sigma, sharpness):
    dt0_v = np.diff(fin, 1, 0)
    dt0_v = np.concatenate((dt0_v, fin[:1, :] - fin[-1:, :]), axis=0)
    dt0_h = np.diff(fin, 1, 1)
    dt0_h = np.concatenate((dt0_h, fin[:, :1] - fin[:, -1:]), axis=1)
    gauker_h = cv2.filter2D(dt0_h, -1, np.ones((1, sigma)), borderType=cv2.BORDER_CONSTANT)
    gauker_v = cv2.filter2D(dt0_v, -1, np.ones((sigma, 1)), borderType=cv2.BORDER_CONSTANT)
    W_h = 1.0 / (abs(gauker_h) * abs(dt0_h) + sharpness)
    W_v = 1.0 / (abs(gauker_v) * abs(dt0_v) + sharpness)
    return W_h, W_v

def convertCol(tmp):
    return np.reshape(tmp.T, (tmp.shape[0] * tmp.shape[1], 1))

def solveLinearEquation(IN, wx, wy, lambd):
    print('IN', IN.shape)
    r, c, ch = IN.shape[0], IN.shape[1], 1
    k = r * c
    dx = -lambd * convertCol(wx)
    dy = -lambd * convertCol(wy)
    tempx = np.concatenate((wx[:, -1:], wx[:, 0:-1]), 1)
    tempy = np.concatenate((wy[-1:, :], wy[0:-1, :]), 0)
    dxa = -lambd * convertCol(tempx)
    dya = -lambd * convertCol(tempy)
    tempx = np.concatenate((wx[:, -1:], np.zeros((r, c - 1))), 1)
    tempy = np.concatenate((wy[-1:, :], np.zeros((r - 1, c))), 0)
    dxd1 = -lambd * convertCol(tempx)
    dyd1 = -lambd * convertCol(tempy)
    wx[:, -1:] = 0
    wy[-1:, :] = 0
    dxd2 = -lambd * convertCol(wx)
    dyd2 = -lambd * convertCol(wy)
    Ax = spdiags(np.concatenate((dxd1, dxd2), 1).T, np.array([-k + r, -r]), k, k)
    Ay = spdiags(np.concatenate((dyd1, dyd2), 1).T, np.array([-r + 1, -1]), k, k)
    D = 1 - (dx + dy + dxa + dya)
    A = (Ax + Ay) + (Ax + Ay).T + spdiags(D.T, np.array([0]), k, k)
    A = A / 1000.0

    matCol = convertCol(IN)
    print('spsolve start', str(datetime.now()))
    OUT = spsolve(A, matCol, permc_spec="MMD_AT_PLUS_A")
    print('spsolve end', str(datetime.now()))
    OUT = OUT / 1000
    OUT = np.reshape(OUT, (c, r)).T
    return OUT

def tsmooth(I, lambd=0.5, sigma=5, sharpness=0.001):
    wx, wy = computeTextureWeights(I, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lambd)
    return S

def rgb2gm(I):
    print('I', I.shape)
    if I.shape[2] and I.shape[2] == 3:
        I = np.power(np.multiply(np.multiply(I[:, :, 0], I[:, :, 1]), I[:, :, 2]), (1.0 / 3))
    return I

def YisBad(Y, isBad):
    return Y[isBad >= 1]
    # Z = []
    # [rows, cols] = Y.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if isBad[i, j] >= 122:
    #             Z.append(Y[i, j])
    # return np.array([Z]).T

def applyK(I, k, a=-0.3293, b=1.1258):
    if not type(I) == 'numpy.ndarray':
        I = np.array(I)
    beta = np.exp((1 - (k ** a)) * b)
    gamma = (k ** a)
    BTF = np.power(I, gamma) * beta
    # try:
    #    BTF = (I ** gamma) * beta
    # except:
    #    print('gamma:', gamma, '---beta:', beta)
    #    BTF = I
    return BTF

def maxEntropyEnhance(I, isBad, mink=1, maxk=10):
    # Y = rgb2gm(np.real(np.maximum(imresize(I, (50, 50), interp='bicubic') / 255.0, 0)))
    Y = np.array(Image.fromarray(I.astype('uint8')).resize((50, 50))) / 255.0
    Y = rgb2gm(Y)
    # Y = rgb2gm(np.real(np.maximum(cv2.resize(I, (50, 50), interpolation=cv2.INTER_LANCZOS4  ), 0)))
    # import matplotlib.pyplot as plt
    # plt.imshow(Y, cmap='gray');plt.show()
    print('isBad', isBad.shape)
    isBad = np.array(Image.fromarray(isBad.astype('uint8')).resize((50, 50), resample=Image.NEAREST))
    print('isBad', isBad.shape)
    # plt.imshow(isBad, cmap='gray');plt.show()
    # Y = YisBad(Y, isBad)
    Y = Y[isBad >= 1]
    # Y = sorted(Y)
    print('-entropy(Y)', -entropy(Y))

    def f(k):
        return -entropy(applyK(Y, k))

    # opt_k = mink
    # k = mink
    # minF = f(k)
    # while k<= maxk:
    #     k+=0.0001
    #     if f(k)<minF:
    #         minF = f(k)
    #         opt_k = k
    opt_k = fminbound(f, mink, maxk)
    print('opt_k:', opt_k)
    # opt_k = 5.363584
    # opt_k = 0.499993757705
    print('opt_k:', opt_k)
    J = applyK(I, opt_k) - 0.01
    return J

def HDR2dark(I, t_our, W):
    W = 1 - W
    I3 = I * W
    isBad = t_our > 0.8
    J3 = maxEntropyEnhance(I, isBad, 0.1, 0.5)
    J3 = J3 * (1 - W)
    fused = I3 + J3
    return I

def oneHDR(I, mu=0.5, a=-0.3293, b=1.1258):
    I = I / 255.0
    t_b = I[:, :, 0]
    for i in range(I.shape[2] - 1):
        t_b = np.maximum(t_b, I[:, :, i + 1])
    # t_b2 = cv2.resize(t_b, (0, 0), fx=0.5, fy=0.5)
    print('t_b', t_b.shape)
    # t_b2 = misc.imresize(t_b, (ceil(t_b.shape[0] / 2), ceil(t_b.shape[1] / 2)),interp='bicubic')
    # print('t_b2', t_b2.shape)
    # t_b2 = t_b / 255.0
    t_b2 = np.array(Image.fromarray(t_b).resize((256, 256)))
    # t_b2 = imresize(t_b, (256, 256), interp='bicubic', mode='F')  # / 255
    t_our = tsmooth(t_b2)
    print('t_our before', t_our.shape)
    t_our = np.array(Image.fromarray(t_our).resize((t_b.shape[1], t_b.shape[0])))
    # t_our = imresize(t_our, t_b.shape, interp='bicubic', mode='F')  # / 255
    print('t_our after', t_our.shape)
    t = np.ndarray(I.shape)
    for ii in range(I.shape[2]):
        t[:, :, ii] = t_our
    print('t', t.shape)
    W = t ** mu
    # cv2.imwrite(filepath + 'W.jpg', W * 255)
    # cv2.imwrite(filepath + '1-W.jpg', (1 - W) * 255)
    # cv2.imwrite(filepath + 't.jpg', t * 255)
    # cv2.imwrite(filepath + '1-t.jpg', (1 - t) * 255)
    print('W', W.shape)
    # isBad = t_our > 0.8
    # I = maxEntropyEnhance(I, isBad)
    I2 = I * W
    isBad = t_our < 0.5
    J = maxEntropyEnhance(I, isBad)
    J2 = J * (1 - W)
    fused = I2 + J2
    # cv2.imwrite(filepath + 'I2.jpg', I2 * 255.0)
    # cv2.imwrite(filepath + 'J2.jpg', J2 * 255.0)
    # fused = HDR2dark(fused, t_our, W)
    return fused
    # return res


def histeq(im, nbr_bins=256):
    warnings.filterwarnings("ignore", category=Warning)
    imhist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def test():
    inputImg = cv2.imread(filepath)
    outputImg = oneHDR(inputImg)
    # outputImg = outputImg * 255.0
    cv2.imwrite(filepath + '1out.jpg', outputImg * 255)
    print("HDR finished")
    print('terminated', str(datetime.now()))
    # cv2.imshow('inputImg', inputImg)
    # cv2.imshow('outputImg', outputImg)
    # print(inputImg.dtype,outputImg.dtype)
    # outputImg = outputImg.astype(int)
    # print(inputImg.dtype, outputImg.dtype)
    # compare = np.concatenate((inputImg,outputImg),axis=1)
    # cv2.imshow('compare', compare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test2():
    image = cv2.imread(filepath)
    for channel in range(image.shape[2]):
        image[:, :, channel], _ = histeq(image[:, :, channel])
    cv2.imwrite("dim1_histeq.png", image)

if __name__ == '__main__':
    print('start', str(datetime.now()))
    test2()
