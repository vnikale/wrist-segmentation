from sklearn.metrics import precision_recall_curve,f1_score,auc,roc_curve
import matplotlib.pyplot as plt
import numpy as np

from ..utils.metrics import dice_coef
from datetime import datetime

from matplotlib.lines import Line2D
import os
from sklearn.metrics import confusion_matrix

import pandas as pd
import re


def aucpr_calc(true_y, pred_y):
    # TODO: refactor this function to descent form

    precision, recall, thresholds = precision_recall_curve(true_y.flatten(),pred_y.flatten())
    # f1 = f1_score(pred_y.flatten(),pred.flatten())

    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    thr_prerec = thresholds[ix]
    print('Best Threshold=%f, F-Score=%f' % (thresholds[ix], fscore[ix]))
    precision2D = precision[ix]
    recall2D = recall[ix]
    aucpr = auc(recall, precision)
    print('AUCPR=%f'%aucpr)

    plt.plot(recall,precision,label='trUnet')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    plt.grid(True)
    plt.xlabel('recall')
    plt.ylabel('precision')
    no_skill = len(true_y.flatten()[true_y.flatten()==1]) / len(true_y.flatten())
    plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
    plt.legend()

    plt.figure()
    fpr, tpr, thresholds = roc_curve(true_y.flatten(),pred_y.flatten())

    gmeans = np.sqrt(tpr * (1-fpr))

    # locate the index of the largest g-mean
    ix = np.argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%f' % (thresholds[ix], gmeans[ix]))

    aucroc = auc(fpr, tpr)
    print('AUCROC=%f'%aucroc)

    plt.plot(fpr, tpr, marker='', label='trUnet')
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True)

    return thr_prerec, aucpr, precision2D, recall2D


def DSC2D(true_y, pred_y,
          model_name,
          dataFold,
          model_path,
          thr_aucpr=0.5):
    # TODO: refactor this function to descent form

    dsc = dice_coef

    N = pred_y.shape[0]
    D = np.arange(0,1.02,0.02)
    n = D.shape[0]
    DSC0 = np.zeros((N,n))
    cartilage0 = np.zeros((N,n))
    for k in range(0,n):
        tr_pred = pred_y > D[k]
        tr_pred = tr_pred.astype(float)
        for i in range(0,N):
            DSC0[i,k] = dsc(tr_pred[i,:,:,0],true_y[i,:,:,0])
            cartilage0[i,k] = np.sum(true_y[i,:,:,0])

    print('Mean DSC: ',np.max(np.mean(DSC0,axis=0)))
    i = np.argmax(np.mean(DSC0,axis=0))

    DSC = np.zeros((N,))
    cartilage = np.zeros((N,))
    # d = D[i]

    if thr_aucpr > 0.9:
        d = D[i]
    else:
        d = thr_aucpr
    DSC = DSC0[:,i]
    cartilage = cartilage0[:,i]
    tr_pred = np.array([pred_y > d], dtype=int)
    tr_pred = np.reshape(tr_pred,(tr_pred.shape[1:]))

    print('Mean 2DDSC: ',(np.mean(DSC,axis=0)))
    print('STD 2DDSC: ',(np.std(DSC,axis=0)))
    tr_pred = tr_pred.astype(float)

    file = model_name + 'mean_DSC_log.txt'
    fp = open(file, 'a')
    fp.write ("\nTime: "+datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fp.write ("\nOpened: %s\nmean DSC: %.5f" % (model_path, np.mean(DSC) ) )
    fp.write ("\nDataset folder: %s\n" % (dataFold))
    fp.write("\nThreshold auc: %g\nThreshold manual %g" % (thr_aucpr,d))

    fp.close()

    return d, cartilage, DSC


def vol_csa_dsc(true_x, true_y,
                pred_y,
                model_name,
                DSC,
                cartilage,
                pat_len,
                res,
                recall2D,
                precision2D,
                model_path,
                folder=r'output\docs\newdata(cv6)\\',
                thr_prerec=0.5):

    #TODO: refactor this function to descent form

    log_fold = folder + model_name
    if not os.path.exists(log_fold):
        os.makedirs(log_fold)

    dsc = dice_coef

    test_mask = true_y
    # d=0.60
    d = thr_prerec
    n = 11
    tr_pred = np.array([pred_y > d], dtype=float)[0, ...]
    # tr_pred = np.reshape(tr_pred,(tr_pred.shape[1:]))

    test_imag0 = np.copy(true_x)

    # test_imag0 -= np.mean(test_imag0)
    # test_imag0 /= np.std(test_imag0)

    test_imag0 -= np.min(test_imag0)
    test_imag0 /= np.max(test_imag0) - np.min(test_imag0)

    img_rgb = test_imag0
    img_rgb = np.append(img_rgb, test_imag0, axis=3)
    img_rgb = np.append(img_rgb, test_imag0, axis=3)
    # img_rgb -= np.min(img_rgb)
    # img_rgb /= np.max(img_rgb) - np.min(img_rgb)
    print(img_rgb.shape, np.max(img_rgb), np.min(img_rgb))

    f1 = plt.figure(figsize=(12, 50))
    a1 = plt.subplot(121)
    a1.set_title('Ground truth')

    img_rgb2 = np.copy(img_rgb)
    img_rgb2[..., 1] = img_rgb[..., 1] + test_mask[..., 0]
    img_rgb2[img_rgb2 > 1] = 1
    plt.imshow(img_rgb2[n, :, :, :])

    a2 = plt.subplot(122)
    a2.set_title('Prediction (DSC = {%.5f})' % dsc(tr_pred[n, :, :, 0], test_mask[n, :, :, 0]))
    img_rgb2[..., 1] = img_rgb[..., 1] + tr_pred[..., 0]
    img_rgb2[img_rgb2 > 1] = 1
    plt.imshow(img_rgb2[n, :, :, :])

    TP = tr_pred * test_mask  # test_mask == 1 and tr_pred == 1
    FP = ((test_mask * 2 - 2) / (-2)) * tr_pred  # test_mask == 0 and tr_pred == 1
    FN = ((tr_pred * 2 - 2) / (-2)) * test_mask  # test_mask == 1 0 and tr_pred == 0
    img_rgb3 = 0.5 * np.copy(img_rgb)
    img_rgb3[..., 1] += np.abs(TP[..., 0])
    img_rgb3[..., 0] += np.abs(FP[..., 0])
    img_rgb3[..., 2] += np.abs(FN[..., 0])
    img_rgb3[img_rgb3 > 1] = 1
    print(np.max(img_rgb3[:, :, 0]), np.min(img_rgb3))
    f2 = plt.figure(figsize=(10, 10))
    legend_elements = [Line2D([0], [0], marker='s', color='w', label='FP',
                              markerfacecolor='r', markersize=15),
                       Line2D([0], [0], marker='s', color='w', label='TP',
                              markerfacecolor='g', markersize=15),
                       Line2D([0], [0], marker='s', color='w', label='FN',
                              markerfacecolor='b', markersize=15)]

    plt.legend(handles=legend_elements, loc='best')
    plt.imshow(img_rgb3[n, :, :, :])

    print('DSC on the full test set: ', np.mean(DSC), '\n')
    args = np.argwhere(np.invert(DSC < 1 - 1e-4))
    print('DSC on the full test set without 0s and 1s: ', np.mean(np.delete(DSC, args)), '\n')

    def VOE_np(y_pred, y_true):
        y_true = y_true.flatten().astype(bool)
        y_pred = y_pred.flatten().astype(bool)
        return 100 * (1. - np.logical_and(y_pred, y_true).astype(float).sum() / (
                    np.logical_or(y_pred, y_true).astype(float).sum() + 1e-3))

    VOE = np.empty((0,))
    AUC = np.empty((0,))

    DSC_pd0 = pd.DataFrame([])
    DSC_3D = np.empty((0,))

    CSA_pred = np.empty((0,))

    CSA_true = np.empty((0,))

    VOL_true = np.empty((0,))
    VOL_pred = np.empty((0,))

    CONF = np.empty((0, 4))
    cart = np.empty((0,))
    car_pd0 = pd.DataFrame([])
    prev = 0
    k = 0
    for i in range(pat_len.shape[0]):
        if pat_len[i] != 0:
            DSC_pd = pd.DataFrame(np.array(DSC[prev:prev + pat_len[i]]), columns=['DSC,Case {0}'.format(i + 1)])
            DSC_pd0 = pd.concat([DSC_pd0, DSC_pd], axis=1)
            #         print(DSC[prev:prev+pat_len[i]])
            f3 = plt.figure(figsize=(10, 5))
            ax1, = plt.plot(range(1, pat_len[i] + 1), DSC[prev:prev + pat_len[i]], 'bo-', label='DSC')
            plt.grid(True)
            DSC_3D = np.append(DSC_3D, dsc(tr_pred[prev:prev + pat_len[i], :, :, 0],
                                           test_mask[prev:prev + pat_len[i], :, :, 0]))
            #         print(DSC_3D,k)

            precision, recall, thresholds = precision_recall_curve(test_mask[prev:prev + pat_len[i], :, :, 0].flatten(),
                                                                   tr_pred[prev:prev + pat_len[i], :, :, 0].flatten())
            aucpr = auc(recall, precision)

            tn, fp, fn, tp = confusion_matrix(test_mask[prev:prev + pat_len[i], :, :, 0].flatten(),
                                              tr_pred[prev:prev + pat_len[i], :, :, 0].flatten()).ravel()
            CONF = np.append(CONF, np.array([[tn], [fp], [fn], [tp]]).transpose(), axis=0)

            VOE = np.append(VOE, VOE_np(tr_pred[prev:prev + pat_len[i], :, :, 0],
                                        test_mask[prev:prev + pat_len[i], :, :, 0]))
            AUC = np.append(AUC, aucpr)

            plt.title('Case {0}, 3D DSC: {1:.4f}'.format(i + 1, DSC_3D[k]))

            k += 1

            plt.xlim([0, pat_len[i] + 1 + 1])
            plt.ylim([0, 1.2])
            plt.xlabel('№ of slice')
            plt.ylabel('DSC, Cartilage')
            maxx = max(cartilage[prev:prev + pat_len[i]])
            ax2, = plt.plot(range(1, pat_len[i] + 1), cartilage[prev:prev + pat_len[i]] / maxx, 'rd--',
                            label='Cartilage tissue')

            car = cartilage[prev:prev + pat_len[i]] / maxx
            csa_i = np.argmax(car)
            csa = tr_pred[prev + csa_i, :, :, 0].sum() * (res[i, 0] * res[i, 1])
            CSA_pred = np.append(CSA_pred, csa)
            csa = test_mask[prev + csa_i, :, :, 0].sum() * (res[i, 0] * res[i, 1])
            CSA_true = np.append(CSA_true, csa)

            vol = test_mask[prev:prev + pat_len[i], :, :, 0].sum() * (res[i, 0] * res[i, 1] * res[i, 2])
            VOL_true = np.append(VOL_true, vol)

            vol = tr_pred[prev:prev + pat_len[i], :, :, 0].sum() * (res[i, 0] * res[i, 1] * res[i, 2])
            VOL_pred = np.append(VOL_pred, vol)

            cart = np.append(cart, car)
            car_pd = pd.DataFrame(np.array(car), columns=['Case {0}'.format(i + 1)])
            car_pd0 = pd.concat([car_pd0, car_pd], axis=1)

            prev += pat_len[i]
            plt.legend(handles=(ax1, ax2))
            figname = 'Patient {0}_vol.svg'.format(i + 1)
            log = os.path.join(log_fold, figname)

            plt.savefig(log, format='svg')

    xlx = 'VOL_true' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    VOL = pd.DataFrame(data=VOL_true)
    VOL.to_excel(log)

    xlx = 'VOL_pred' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    VOL = pd.DataFrame(data=VOL_pred)
    VOL.to_excel(log)

    xlx = 'confusssion_matrix' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    mat = pd.DataFrame(data=CONF, columns=['tn', 'fp', 'fn', 'tp'])
    mat.to_excel(log)

    s = re.findall('/(\w+\W\w+)/', model_path)
    # DSC_pd0.to_excel('DSC_unet_al_{0}_{1}.xlsx'.format(s,'dataFold'))
    xlx = 'DSC_' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    DSC_pd0.to_excel(log)

    xlx = 'CSA_true' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    CSA = pd.DataFrame(data=CSA_true)
    CSA.to_excel(log)

    xlx = 'CSA_pred' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    CSA = pd.DataFrame(data=CSA_pred)
    CSA.to_excel(log)

    xlx = 'cartilage' + '.xlsx'
    log = os.path.join(os.path.dirname(log_fold), xlx)
    CAR = pd.DataFrame(data=cart)
    CAR.to_excel(log)

    xlx = '3D_DSC' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    DSC = pd.DataFrame(data=DSC_3D)
    DSC.to_excel(log)

    xlx = 'VOE' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    VOEc = pd.DataFrame(data=VOE)
    VOEc.to_excel(log)

    xlx = '3D_AUC' + model_name + '.xlsx'
    log = os.path.join(log_fold, xlx)
    AUCc = pd.DataFrame(data=AUC)
    AUCc.to_excel(log)

    xlx = 'cartilage_cases' + '.xlsx'
    log = os.path.join(os.path.dirname(log_fold), xlx)
    car_pd0.to_excel(log)

    print('Mean 3D DSC: ', np.mean(DSC_3D))
    print('Median 3D DSC: ', np.median(DSC_3D))
    print('25% 3D DSC: ', np.quantile(DSC_3D, 0.25))
    print('75% 3D DSC: ', np.quantile(DSC_3D, 0.75))
    print('STD 3D DSC: ', np.std(DSC_3D))

    print('Mean VoE: ', np.mean(VOE))
    print('STD VoE: ', np.std(VOE))

    print('Median  VOE: ', np.median(VOE))
    print('25%  VOE: ', np.quantile(VOE, 0.25))
    print('75%  VOE: ', np.quantile(VOE, 0.75))

    print('Mean AUC: ', np.mean(AUC))
    print('STD AUC: ', np.std(AUC))
    print('Median  AUC: ', np.median(AUC))
    print('25%  AUC: ', np.quantile(AUC, 0.25))
    print('75%  AUC: ', np.quantile(AUC, 0.75))

    print('FP ', FP.sum())
    print('Precision ', precision2D)
    print('Recall ', recall2D)

    file = os.path.join(log_fold, 'mean_DSC_log.txt')
    fp = open(file, 'w')
    fp.write("\nTime: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fp.write("\nMean auc: %g\nstd auc %g" % (np.mean(AUC), np.std(AUC)))
    fp.write("\nMean 3D DSC: %g\nstd 3D DSC %g" % (np.mean(DSC_3D), np.std(DSC_3D)))
    fp.write("\nMedian 3D DSC: %g\n0.25 3D DSC %g\n0.75 3D DSC %g" % (
    np.median(DSC_3D), np.quantile(DSC_3D, 0.25), np.quantile(DSC_3D, 0.75)))
    fp.write("\nMedian 3D VOE: %g\n0.25 3D VOE %g\n0.75 3D VOE %g" % (
    np.median(VOE), np.quantile(VOE, 0.25), np.quantile(VOE, 0.75)))
    fp.write("\nMedian 3D AUC: %g\n0.25 3D AUC %g\n0.75 3D AUC %g" % (
    np.median(AUC), np.quantile(AUC, 0.25), np.quantile(AUC, 0.75)))
    fp.write("\nMean VOE: %g\nstd DSC %g" % (np.mean(VOE), np.std(VOE)))
    fp.write("\nFP: %g" % (FP.sum()))
    fp.write("\nPrecision: %g" % (precision2D))
    fp.write("\nRecall: %g" % (recall2D))
    fp.close()

    return DSC_3D