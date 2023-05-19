import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import pickle
import random
import numpy as np
# from networks.efficientnetb0 import EfficientNetY1
from networks.mbgru import EfficientGRUY0
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


class MyDataset(data.Dataset):
    def __init__(self, in_data, labels):
        self.data = in_data
        self.labels = labels

    def __getitem__(self, item):
        st_map, label = self.data[item], self.labels[item]
        return st_map, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    lr_orig = 0.1
    batch_size = 200
    epoches = 16
    DEV_BATCH_SIZE = 91
    TEST_BATCH_SIZE = 10
    train_mode = "STN"
    start_round = 1
    if len(sys.argv) > 1:
        train_mode = sys.argv[1]
        start_round = int(sys.argv[2])
    method_name = "3DMAD_HI_S_ALL1S_BIO1"
    feature_type = 'bp'
    model_dir = "./models/"
    num_subjects = 17

    Num_rows = 4
    Num_cols = 6
    Num_LOOCV = 20
    Num_thresholds = 200

    random.seed(42)

    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
        os.mkdir('./checkpoint/{}_{}'.format(method_name, train_mode))
    elif not os.path.isdir('./checkpoint/{}_{}'.format(method_name, train_mode)):
        os.mkdir('./checkpoint/{}_{}'.format(method_name, train_mode))
    for round_id in range(start_round, Num_LOOCV + start_round):
        print('Round {}'.format(round_id))
        if not os.path.isdir('./checkpoint/{}_{}/round_{}'.format(method_name, train_mode, round_id)):
            os.mkdir('./checkpoint/{}_{}/round_{}'.format(method_name, train_mode, round_id))
        else:
            print("Warning: Non-empty folder!")
            os._exit(1)
        data_split_list = []
        for test_id in range(1, num_subjects + 1):
            x = [i for i in range(1, num_subjects + 1)]
            x.remove(test_id)
            random.shuffle(x)
            train_subject = x[:8]
            development_subject = x[8:]
            data_split_list.append([test_id, train_subject, development_subject])
            print("train subjects: {}".format(train_subject))
            print("development subjects: {}".format(development_subject))
            print("test subject: {}".format(test_id))
            with open("./checkpoint/{}_{}/train.log".format(method_name, train_mode), "a") as f:
                f.write("{} Round {}\n".format(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()), round_id))
                f.write("lr: {:.3f}, batch size: {}, epochs: {}, config: {}\n".format(
                    lr_orig, batch_size, epoches, method_name))
                f.write("train subjects: {}\ndevelopment subjects: {}\ntest subject: {}\n\n".format(
                    train_subject, development_subject, test_id))

            data_train_real = None
            data_development_real = None
            data_train_attack = None
            data_development_attack = None
            data_test_real = []
            data_test_attack = []
            for i in train_subject:
                with open("{}{}/3DMAD_real_{}_{}_{}_{}_{}.bio.pkl".format(
                        model_dir, method_name, Num_rows, Num_cols, i, train_mode, feature_type), "rb") as f:
                    if data_train_real is None:
                        data_train_real = pickle.load(f).astype('float32')
                    else:
                        data_train_real = np.concatenate((data_train_real, pickle.load(f).astype('float32')), axis=0)
                with open("{}{}/3DMAD_attack_{}_{}_{}_{}_{}.bio.pkl".format(
                        model_dir, method_name, Num_rows, Num_cols, i, train_mode, feature_type), "rb") as f:
                    if data_train_attack is None:
                        data_train_attack = pickle.load(f).astype('float32')
                    else:
                        data_train_attack = np.concatenate((data_train_attack, pickle.load(f).astype('float32')), axis=0)
            print(data_train_real.shape, data_train_attack.shape)
            for i in development_subject:
                with open("{}{}/3DMAD_real_{}_{}_{}_{}_{}.bio.pkl".format(
                        model_dir, method_name, Num_rows, Num_cols, i, train_mode, feature_type), "rb") as f:
                    if data_development_real is None:
                        data_development_real = pickle.load(f).astype('float32')
                    else:
                        data_development_real = np.concatenate(
                            (data_development_real, pickle.load(f).astype('float32')), axis=0)
                with open("{}{}/3DMAD_attack_{}_{}_{}_{}_{}.bio.pkl".format(
                        model_dir, method_name, Num_rows, Num_cols, i, train_mode, feature_type), "rb") as f:
                    if data_development_attack is None:
                        data_development_attack = pickle.load(f).astype('float32')
                    else:
                        data_development_attack = np.concatenate(
                            (data_development_attack, pickle.load(f).astype('float32')), axis=0)
            print(data_development_real.shape, data_development_attack.shape)
            with open("{}{}/3DMAD_real_{}_{}_{}_{}_{}.bio.pkl".format(
                    model_dir, method_name, Num_rows, Num_cols, test_id, train_mode, feature_type), "rb") as f:
                data_test_real_all = pickle.load(f).astype('float32')
            for i in range(data_test_real_all.shape[0]):
                j = i % 91
                if j % 10 == 0:
                    data_test_real.append(data_test_real_all[j])
            data_test_real = np.array(data_test_real)
            print(data_test_real.shape)
            with open("{}{}/3DMAD_attack_{}_{}_{}_{}_{}.bio.pkl".format(
                    model_dir, method_name, Num_rows, Num_cols, test_id, train_mode, feature_type), "rb") as f:
                data_test_attack_all = pickle.load(f).astype('float32')
            for i in range(data_test_attack_all.shape[0]):
                j = i % 91
                if j % 10 == 0:
                    data_test_attack.append(data_test_attack_all[j])
            data_test_attack = np.array(data_test_attack)
            print(data_test_attack.shape)

            train_data_positive = data_train_real.transpose(0, 3, 1, 2)
            train_data_negative = data_train_attack.transpose(0, 3, 1, 2)
            train_labels = [1 for i in range(data_train_real.shape[0])] + [0 for i in range(data_train_attack.shape[0])]

            development_data_positive = data_development_real.transpose(0, 3, 1, 2)
            development_data_negative = data_development_attack.transpose(0, 3, 1, 2)
            development_labels = [1 for i in range(data_development_real.shape[0])] + \
                                 [0 for i in range(data_development_attack.shape[0])]

            test_data_positive = data_test_real.transpose(0, 3, 1, 2)
            test_data_negative = data_test_attack.transpose(0, 3, 1, 2)
            test_labels = [1 for i in range(data_test_real.shape[0])] + [0 for i in range(data_test_attack.shape[0])]

            if torch.cuda.device_count() >= 2:
                num_workers = 2
            else:
                num_workers = 0
            trainset = MyDataset(np.concatenate((train_data_positive, train_data_negative)), train_labels)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                      num_workers=num_workers)
            developmentset = MyDataset(np.concatenate((development_data_positive, development_data_negative)),
                                       development_labels)
            developmentloader = torch.utils.data.DataLoader(
                developmentset, batch_size=DEV_BATCH_SIZE, shuffle=False, num_workers=num_workers)
            testset = MyDataset(np.concatenate((test_data_positive, test_data_negative)), test_labels)
            testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                                     num_workers=num_workers)
            print("trainset data shape: {}".format(trainset.data.shape))
            print("developmentset data shape: {}".format(developmentset.data.shape))
            print("testset data shape: {}".format(testset.data.shape))

            device = "cuda" if torch.cuda.is_available() else "cpu"
            net = EfficientGRUY0(num_classes=2, in_channels=18)
            net = net.to(device)
            if device == 'cuda':
                net = nn.DataParallel(net)
                cudnn.benchmark = True
            else:
                print("CPU training not supported")
                os._exit(1)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=lr_orig, momentum=0.9, weight_decay=5e-3)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

            best_acc = 0
            start_epoch = 0
            total_time = 0
            for epoch in range(start_epoch, start_epoch + epoches):
                print("Epoch: {}".format(epoch))

                net.train()
                train_loss = 0
                correct = 0
                total = 0
                batches = len(trainloader)
                start_time = time.time()
                for batch_idx, (train_X, train_y) in enumerate(trainloader):
                    train_X = train_X.to(device)
                    train_y = train_y.to(device)
                    optimizer.zero_grad()
                    outputs = net(train_X)
                    loss = criterion(outputs, train_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predict_y = outputs.max(1)
                    total += train_y.size(0)
                    correct += predict_y.eq(train_y).sum().item()
                    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

                train_acc = 100. * correct / total
                end_time = time.time()
                total_time += end_time - start_time

                results_report = []
                net.eval()
                test_loss = 0
                batches = len(developmentloader)
                y_true = []
                y_pred = []
                dev_scores = []
                TP = np.zeros(Num_thresholds, dtype='int')
                FP = np.zeros(Num_thresholds, dtype='int')
                TN = np.zeros(Num_thresholds, dtype='int')
                FN = np.zeros(Num_thresholds, dtype='int')
                TPR = np.zeros(Num_thresholds)
                FPR = np.zeros(Num_thresholds)
                FAR = np.zeros(Num_thresholds)
                FRR = np.zeros(Num_thresholds)
                TP_m = np.zeros(Num_thresholds, dtype='int')
                FP_m = np.zeros(Num_thresholds, dtype='int')
                TN_m = np.zeros(Num_thresholds, dtype='int')
                FN_m = np.zeros(Num_thresholds, dtype='int')
                TPR_m = np.zeros(Num_thresholds)
                FPR_m = np.zeros(Num_thresholds)
                FAR_m = np.zeros(Num_thresholds)
                FRR_m = np.zeros(Num_thresholds)
                with torch.no_grad():
                    for batch_idx, (test_X, test_y) in enumerate(developmentloader):
                        y_true.extend(list(test_y))
                        test_X = test_X.to(device)
                        test_y = test_y.to(device)
                        outputs = net(test_X)
                        loss = criterion(outputs, test_y)
                        test_loss += loss.item()
                        _, predict_y = outputs.max(1)

                        outputs_np = outputs.cpu().detach().numpy()
                        confidences = 1 / (1 + np.exp(-0.5 * outputs_np))
                        confidences = confidences[:, 1] / confidences.sum(axis=1)
                        y_pred.extend(list(confidences))
                        ground_truth = test_y.cpu().numpy()
                        batch_scores = np.zeros((ground_truth.shape[0], 4))
                        batch_scores[:, :2] = outputs_np
                        batch_scores[:, 2] = confidences
                        batch_scores[:, 3] = ground_truth
                        dev_scores.append(batch_scores)

                y_true = np.array(y_true)
                y_pred = np.array(y_pred)             
                dev_scores = np.concatenate(dev_scores, axis=0)

                ordered_confidences = list(y_pred)
                ordered_confidences.sort()
                thresholds = []
                for ratio in range(Num_thresholds):
                    threshold_idx = int(1.0 * ratio * len(ordered_confidences) / Num_thresholds)
                    thresholds.append(ordered_confidences[threshold_idx])
                
                num_batches = len(ordered_confidences) // DEV_BATCH_SIZE
                confidence_rates = np.resize(y_pred, (num_batches, DEV_BATCH_SIZE))
                labels = np.resize(y_true, (num_batches, DEV_BATCH_SIZE))

                for batch_idx in range(num_batches):
                    confidences = confidence_rates[batch_idx]
                    ground_truth = labels[batch_idx]
                    for ite in range(Num_thresholds):
                        threshold = thresholds[ite]
                        local_TP, local_FP, local_FN, local_TN = 0, 0, 0, 0
                        for i in range(DEV_BATCH_SIZE):
                            if confidences[i] > threshold and ground_truth[i] == 1:
                                local_TP += 1
                                TP[ite] += 1
                            elif confidences[i] > threshold and ground_truth[i] == 0:
                                local_FP += 1
                                FP[ite] += 1
                            elif confidences[i] <= threshold and ground_truth[i] == 1:
                                local_FN += 1
                                FN[ite] += 1
                            else:
                                local_TN += 1
                                TN[ite] += 1
                        if local_TP >= local_FP and local_TP >= local_FN and local_TP >= local_TN:
                            TP_m[ite] += 1
                        if local_TN >= local_TP and local_TN >= local_FP and local_TN >= local_FN:
                            TN_m[ite] += 1
                        if local_FP >= local_TP and local_FP >= local_FN and local_FP >= local_TN:
                            FP_m[ite] += 1
                        if local_FN >= local_TP and local_FN >= local_FP and local_FN >= local_TN:
                            FN_m[ite] += 1
                for ite in range(Num_thresholds):
                    TPR[ite] = TP[ite] / (TP[ite] + FN[ite])
                    FPR[ite] = FP[ite] / (TN[ite] + FP[ite])
                    FAR[ite] = FP[ite] / (FP[ite] + TN[ite])
                    FRR[ite] = FN[ite] / (TP[ite] + FN[ite])
                    TPR_m[ite] = TP_m[ite] / (TP_m[ite] + FN_m[ite])
                    FPR_m[ite] = FP_m[ite] / (TN_m[ite] + FP_m[ite])
                    FAR_m[ite] = FP_m[ite] / (FP_m[ite] + TN_m[ite])
                    FRR_m[ite] = FN_m[ite] / (TP_m[ite] + FN_m[ite])

                    # print(ite, TP[ite], FP[ite], TN[ite], FN[ite], TPR_m[ite], FPR_m[ite], FAR_m[ite], FRR_m[ite],
                    #       TPR[ite], FPR[ite], FAR[ite], FRR[ite])
                    results_report.append(
                        [TP[ite], TN[ite], FP[ite], FN[ite], TPR_m[ite], FPR_m[ite], FAR_m[ite], FRR_m[ite], TPR[ite],
                         FPR[ite], FAR[ite], FRR[ite]])

                # save evaluation results
                with open("./checkpoint/{}_{}/round_{}/devlog_{}_cv{}_ep{}.csv".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "w") as f:
                    for i in results_report:
                        f.write("{:d},{:d},{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]))
                with open("./checkpoint/{}_{}/round_{}/devlog_{}_cv{}_ep{}.pkl".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "wb") as f:
                    pickle.dump(results_report, f)
                with open("./checkpoint/{}_{}/round_{}/devscore_{}_cv{}_ep{}.csv".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "w") as f:
                    for i in dev_scores:
                        f.write("{},{},{},{},\n".format(i[0], i[1], i[2], i[3]))

                results_report = []
                net.eval()
                test_loss = 0
                batches = len(testloader)
                y_true = []
                y_pred = []
                test_scores = []
                TP = np.zeros(Num_thresholds, dtype='int')
                FP = np.zeros(Num_thresholds, dtype='int')
                TN = np.zeros(Num_thresholds, dtype='int')
                FN = np.zeros(Num_thresholds, dtype='int')
                TPR = np.zeros(Num_thresholds)
                FPR = np.zeros(Num_thresholds)
                FAR = np.zeros(Num_thresholds)
                FRR = np.zeros(Num_thresholds)
                TP_m = np.zeros(Num_thresholds, dtype='int')
                FP_m = np.zeros(Num_thresholds, dtype='int')
                TN_m = np.zeros(Num_thresholds, dtype='int')
                FN_m = np.zeros(Num_thresholds, dtype='int')
                TPR_m = np.zeros(Num_thresholds)
                FPR_m = np.zeros(Num_thresholds)
                FAR_m = np.zeros(Num_thresholds)
                FRR_m = np.zeros(Num_thresholds)
                with torch.no_grad():
                    for batch_idx, (test_X, test_y) in enumerate(testloader):
                        y_true.extend(list(test_y))
                        test_X = test_X.to(device)
                        test_y = test_y.to(device)
                        outputs = net(test_X)
                        loss = criterion(outputs, test_y)
                        test_loss += loss.item()
                        _, predict_y = outputs.max(1)

                        outputs_np = outputs.cpu().detach().numpy()
                        confidences = 1 / (1 + np.exp(-0.5 * outputs_np))
                        confidences = confidences[:, 1] / confidences.sum(axis=1)
                        y_pred.extend(list(confidences))
                        ground_truth = test_y.cpu().numpy()
                        batch_scores = np.zeros((ground_truth.shape[0], 4))
                        batch_scores[:, :2] = outputs_np
                        batch_scores[:, 2] = confidences
                        batch_scores[:, 3] = ground_truth
                        test_scores.append(batch_scores)
                
                y_true = np.array(y_true)
                y_pred = np.array(y_pred)
                test_scores = np.concatenate(test_scores, axis=0)

                num_batches = y_pred.shape[0] // TEST_BATCH_SIZE
                confidence_rates = np.resize(y_pred, (num_batches, TEST_BATCH_SIZE))
                labels = np.resize(y_true, (num_batches, TEST_BATCH_SIZE))

                for batch_idx in range(num_batches):
                    confidences = confidence_rates[batch_idx]
                    ground_truth = labels[batch_idx]
                    for ite in range(Num_thresholds):
                        threshold = thresholds[ite]
                        local_TP, local_FP, local_FN, local_TN = 0, 0, 0, 0
                        for i in range(TEST_BATCH_SIZE):
                            if confidences[i] > threshold and ground_truth[i] == 1:
                                local_TP += 1
                                TP[ite] += 1
                            elif confidences[i] > threshold and ground_truth[i] == 0:
                                local_FP += 1
                                FP[ite] += 1
                            elif confidences[i] <= threshold and ground_truth[i] == 1:
                                local_FN += 1
                                FN[ite] += 1
                            else:
                                local_TN += 1
                                TN[ite] += 1
                        if local_TP >= local_FP and local_TP >= local_FN and local_TP >= local_TN:
                            TP_m[ite] += 1
                        if local_TN >= local_TP and local_TN >= local_FP and local_TN >= local_FN:
                            TN_m[ite] += 1
                        if local_FP >= local_TP and local_FP >= local_FN and local_FP >= local_TN:
                            FP_m[ite] += 1
                        if local_FN >= local_TP and local_FN >= local_FP and local_FN >= local_TN:
                            FN_m[ite] += 1
                for ite in range(Num_thresholds):
                    TPR[ite] = TP[ite] / (TP[ite] + FN[ite])
                    FPR[ite] = FP[ite] / (TN[ite] + FP[ite])
                    FAR[ite] = FP[ite] / (FP[ite] + TN[ite])
                    FRR[ite] = FN[ite] / (TP[ite] + FN[ite])
                    TPR_m[ite] = TP_m[ite] / (TP_m[ite] + FN_m[ite])
                    FPR_m[ite] = FP_m[ite] / (TN_m[ite] + FP_m[ite])
                    FAR_m[ite] = FP_m[ite] / (FP_m[ite] + TN_m[ite])
                    FRR_m[ite] = FN_m[ite] / (TP_m[ite] + FN_m[ite])

                    # print(ite, TP[ite], FP[ite], TN[ite], FN[ite], TPR_m[ite], FPR_m[ite], FAR_m[ite], FRR_m[ite],
                    #       TPR[ite], FPR[ite], FAR[ite], FRR[ite])
                    results_report.append(
                        [TP[ite], TN[ite], FP[ite], FN[ite], TPR_m[ite], FPR_m[ite], FAR_m[ite], FRR_m[ite], TPR[ite],
                         FPR[ite], FAR[ite], FRR[ite]])

                # save evaluation results
                with open("./checkpoint/{}_{}/round_{}/testlog_{}_cv{}_it{}.csv".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "w") as f:
                    for i in results_report:
                        f.write("{:d},{:d},{:d},{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                            i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10], i[11]))
                with open("./checkpoint/{}_{}/round_{}/testlog_{}_cv{}_it{}.pkl".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "wb") as f:
                    pickle.dump(results_report, f)
                with open("./checkpoint/{}_{}/round_{}/testscore_{}_cv{}_it{}.csv".format(
                        method_name, train_mode, round_id, feature_type, test_id, epoch), "w") as f:
                    for i in test_scores:
                        f.write("{},{},{},{},\n".format(i[0], i[1], i[2], i[3]))

                print("save new model")
                state = {
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'runtime': total_time,
                    'train_acc': train_acc,
                }
                torch.save(state, './checkpoint/{}_{}/round_{}/{}_cv{}_it{}.pth'.format(
                    method_name, train_mode, round_id, feature_type, test_id, epoch))

                scheduler.step()

        with open("./checkpoint/{}_{}/round_{}/train.pkl".format(method_name, train_mode, round_id), "wb") as f:
            pickle.dump(data_split_list, f)
