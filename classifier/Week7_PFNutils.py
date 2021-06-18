#imports
import uproot as ur
import awkward as ak
import numpy as np

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2" #specify GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# energyflow imports
import energyflow as ef
from energyflow.archs import PFN
from energyflow.utils import data_split

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import graph_util as gu
import plot_util as pu

from scipy.interpolate import interp1d

def GlobalModel(X_train, X_val, X_test, Y_train, Y_val, Y_test, X_glob_tr, X_glob_val, X_glob_te, epochs, batch_size, num_glob, filename, fsize):
    #for now, try all three global features in X_glob okay?! 
    #one function to run the model & create the metrics
    
    Phi_sizes, F_sizes = (100, 100, 128), (fsize, fsize, fsize) #F affects the global features, so try increasing?
    # F initially was (100,100,100)
   
    #concatenate the Xs (needed for global features):
    X_tr = [X_train, X_glob_tr] #will this work?
    X_vali = [X_val, X_glob_val]
    X_te = [X_test, X_glob_te]
    
    #make the model:
    pfn = PFN(input_dim=X_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes, num_global_features =num_glob)
    
    #try callbacks:
    callback = tf.keras.callbacks.ModelCheckpoint(data_path+filename, save_best_only=True)
    
    # train model
    history = pfn.fit(X_tr, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_vali, Y_val), verbose=1, callbacks=[callback])
   
    return history

def metrics(model, X_te, X_globte, select):
    #make metrics
    X_1, X_2 = X_te, X_globte
    fps = []
    tps = []
    threshss = []
    aucs = []
    for selection in select:
        X_interm = [X_1[selection], X_2[selection]] #YAY got it working!! just had to split it up to make the selection
        preds = model.predict(X_interm, batch_size=1000) 
        pfn_fp, pfn_tp, threshs = roc_curve(Y_test[selection][:,1], preds[:,1])
        
        fps.append(pfn_fp)
        tps.append(pfn_tp)
        threshss.append(threshs)
        
        # get area under the ROC curve
        auc = roc_auc_score(Y_test[selection][:,1], preds[:,1])
        aucs.append(auc)
        print('PFN AUC:', auc)
        
    return fps, tps, aucs

def interp95(fps, tps):
    fg = []
    for i in range(len(fps)):
        fg.append(interp1d(tps[i], 1/fps[i]))

    reg95 = []
    for i in range(len(fg)):
        reg95.append(fg[i](.95))
    return reg95

def plots1(ranges, aucs, aucs_ng, reg95, reg95_ng, rangename='eta', globalpars='eta, Pt, E', logx=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12,6])
    if logx:
        ax1.semilogx()
    ax1.set_xlim(0., max(ranges)+.1)
    ax1.plot(ranges, aucs, linewidth=3, c='xkcd:sandstone', marker='o', label='global '+globalpars)
    ax1.plot(ranges, aucs_ng, linewidth=3, c='xkcd:macaroni and cheese', marker='o', label='no global features')
    ax1.set_xlabel('absolute range in '+rangename)
    ax1.set_ylabel('AUC')
    ax1.legend()
    
    if logx:
        ax2.semilogx()
    ax2.semilogy()
    ax2.set_xlim(0., max(ranges)+.1)
    ax2.plot(ranges, reg95, linewidth=3, c='xkcd:dark sand', marker='o', label='global '+globalpars)
    ax2.plot(ranges, reg95_ng, linewidth=3, c='xkcd:pale peach', marker='o', label='no global features')
    ax2.set_xlabel('absolute range in '+rangename)
    ax2.set_ylabel('Rejection at 95% efficiency')
    ax2.legend()
    
def histplots(history):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=[12,6])
    ax1.set_xlim(0,len(history.history['acc']))
    ax1.plot(history.history['acc'], label='training set', c='xkcd:yellow orange', linewidth=3)
    ax1.plot(history.history['val_acc'], label='test set', c='xkcd:desert', linewidth=3)
    ax1.set_title('model accuracy')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('epoch')
    ax1.legend()

    ax2.set_xlim(0,len(history.history['acc']))
    ax2.plot(history.history['loss'], label='training set', c='xkcd:orangey yellow', linewidth=3)
    ax2.plot(history.history['val_loss'], label='test set', c='xkcd:bronze', linewidth=3)
    ax2.set_title('model loss')
    ax2.set_ylabel('loss')
    ax2.set_xlabel('epoch')
    ax2.set_yscale('log')
    ax2.legend()
    
def make_ranges(eta_test, ET_test, Pt_test, Eng_test):
    # define ranges and selections in eta

    eta_ranges = np.arange(0., 3.1, .1)+.05

    selec_01 = abs(eta_test) < 0.1
    selec_02 = (abs(eta_test) >= .1) & (abs(eta_test) < .2)
    selec_03 = (abs(eta_test) >= .2) & (abs(eta_test) < .3)
    selec_04 = (abs(eta_test) >= .3) & (abs(eta_test) < .4)
    selec_05 = (abs(eta_test) >= .4) & (abs(eta_test) < .5)
    selec_06 = (abs(eta_test) >= .5) & (abs(eta_test) < .6)
    selec_07 = (abs(eta_test) >= .6) & (abs(eta_test) < .7)
    selec_08 = (abs(eta_test) >= .7) & (abs(eta_test) < .8)
    selec_09 = (abs(eta_test) >= .8) & (abs(eta_test) < .9)
    selec_10 = (abs(eta_test) >= .9) & (abs(eta_test) < 1.)
    selec_11 = (abs(eta_test) >= 1.) & (abs(eta_test) < 1.1)
    selec_12 = (abs(eta_test) >= 1.1) & (abs(eta_test) < 1.2)
    selec_13 = (abs(eta_test) >= 1.2) & (abs(eta_test) < 1.3)
    selec_14 = (abs(eta_test) >= 1.3) & (abs(eta_test) < 1.4)
    selec_15 = (abs(eta_test) >= 1.4) & (abs(eta_test) < 1.5)
    selec_16 = (abs(eta_test) >= 1.5) & (abs(eta_test) < 1.6)
    selec_17 = (abs(eta_test) >= 1.6) & (abs(eta_test) < 1.7)
    selec_18 = (abs(eta_test) >= 1.7) & (abs(eta_test) < 1.8)
    selec_19 = (abs(eta_test) >= 1.8) & (abs(eta_test) < 1.9)
    selec_20 = (abs(eta_test) >= 1.9) & (abs(eta_test) < 2.)
    selec_21 = (abs(eta_test) >= 2.) & (abs(eta_test) < 2.1)
    selec_22 = (abs(eta_test) >= 2.1) & (abs(eta_test) < 2.2)
    selec_23 = (abs(eta_test) >= 2.2) & (abs(eta_test) < 2.3)
    selec_24 = (abs(eta_test) >= 2.3) & (abs(eta_test) < 2.4)
    selec_25 = (abs(eta_test) >= 2.4) & (abs(eta_test) < 2.5)
    selec_26 = (abs(eta_test) >= 2.5) & (abs(eta_test) < 2.6)
    selec_27 = (abs(eta_test) >= 2.6) & (abs(eta_test) < 2.7)
    selec_28 = (abs(eta_test) >= 2.7) & (abs(eta_test) < 2.8)
    selec_29 = (abs(eta_test) >= 2.8) & (abs(eta_test) < 2.9)
    selec_30 = (abs(eta_test) >= 2.9) & (abs(eta_test) < 3.)
    selec_all = abs(eta_test) <= 3. 

    eta_sel = [selec_01, selec_02, selec_03, selec_04, selec_05, selec_06, selec_07, selec_08, selec_09, selec_10,
                  selec_11, selec_12, selec_13, selec_14, selec_15, selec_16, selec_17, selec_18, selec_19, selec_20,
                  selec_21, selec_22, selec_23, selec_24, selec_25, selec_26, selec_27, selec_28, selec_29, selec_30,
                  selec_all]
    
    #define ranges and selections in ET:
    ET_range = np.logspace(np.log10(min(ET_test)), np.log10(max(ET_test)+1), 30)

    ET_sel = [abs(ET_test) < ET_range[1]]
    for i in range(1, len(ET_range)):
        selec_ = (abs(ET_test) >= ET_range[i-1]) & (abs(ET_test) < ET_range[i])
        ET_sel.append(selec_)

    ET_sel.append(abs(ET_test) < ET_range[-1]+1) #last one includes all

    #define ranges and selections in Pt:
    Pt_range = np.logspace(np.log10(min(Pt_test)), np.log10(max(Pt_test)+1), 30)

    Pt_sel = [abs(Pt_test) < Pt_range[1]]
    for i in range(1, len(Pt_range)):
        selec_ = (abs(Pt_test) >= Pt_range[i-1]) & (abs(Pt_test) < Pt_range[i])
        Pt_sel.append(selec_)

    Pt_sel.append(abs(Pt_test) < Pt_range[-1]+1) #last element includes all
    
    #define ranges and selections in E:

    E_range = np.logspace(np.log10(min(Eng_test)), np.log10(max(Eng_test)+1), 30)

    E_sel = [abs(Eng_test) < E_range[1]]
    for i in range(1, len(E_range)):
        selec_ = (abs(Eng_test) >= E_range[i-1]) & (abs(Eng_test) < E_range[i])
        E_sel.append(selec_)

    E_sel.append(abs(Eng_test) < E_range[-1]+1)
    
    return eta_ranges, eta_sel, ET_range, ET_sel, Pt_range, Pt_sel, E_range, E_sel

def plothelper(ax, labs, fps, tps, aucs, col, fps_all, tps_all, aucs_all):
    ax.set_xlim(0.,.2)
    ax.set_ylim(.8,1.)
    ax.plot([0, 1], [0, 1], 'k--')
    for i in range(len(labs)):
        ax.plot(fps[i], tps[i], c=col[i], linewidth=3, label=labs[i] + ' AUC = %.3f '%aucs[i])
    ax.plot(fps_all, tps_all, c='b', label='all'+' AUC = %.3f '%aucs_all)
    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    ax.legend() 
    
def AUCplot(rang, fps, tps, auc, rangetype='eta'):
    #rangetype can be 'eta', 'E_T', 'E', or 'Pt'
    if rangetype == 'eta':
        labs1 = ['$\eta$ < .1', '.1 <= $\eta$ < .2', '.2 <= $\eta$ < .3','.3 <= $\eta$ < .4','.4 <= $\eta$ < .5']
        labs2 = ['.5 <= $\eta$ < .6','.6 <= $\eta$ < .7','.7 <= $\eta$ < .8','.8 <= $\eta$ < .9','.9 <= $\eta$ < 1.']
        labs3 = ['1. <= $\eta$ < 1.1','1.1 <= $\eta$ < 1.2','1.2 <= $\eta$ < 1.3','1.3 <= $\eta$ < 1.4','1.4 <= $\eta$ < 1.5']
        labs4 = ['1.5 <= $\eta$ < 1.6','1.6 <= $\eta$ < 1.7','1.7 <= $\eta$ < 1.8','1.8 <= $\eta$ < 1.9','1.9 <= $\eta$ < 2']
        labs5 = ['2. <= $\eta$ < 2.1','2.1 <= $\eta$ < 2.2','2.2 <= $\eta$ < 2.3','2.3 <= $\eta$ < 2.4','2.4 <= $\eta$ < 2.5']
        labs6 = ['2.5 <= $\eta$ < 2.6','2.6 <= $\eta$ < 2.7','2.7 <= $\eta$ < 2.8','2.8 <= $\eta$ < 2.9','2.9 <= $\eta$ < 3.']
    elif rangetype=='E_T':
        labs1 = ['$E_T$ < %.3f'%rang[0], '%.3f <= $E_T$ < %.3f'%(rang[0], rang[1]), '%.3f <= $E_T$ < %.3f'%(rang[1], rang[2]),'%.3f <= $E_T$ < %.3f'%(rang[2], rang[3]),'%.3f <= $E_T$ < %.3f'%(rang[3], rang[4])]
        labs2 = ['%.3f <= $E_T$ < %.3f'%(rang[4], rang[5]), '%.3f <= $E_T$ < %.3f'%(rang[5], rang[6]), '%.3f <= $E_T$ < %.3f'%(rang[6], rang[7]),'%.3f <= $E_T$ < %.3f'%(rang[7], rang[8]),'%.3f <= $E_T$ < %.3f'%(rang[8], rang[9])]
        labs3 = ['%.3f <= $E_T$ < %.3f'%(rang[9], rang[10]), '%.3f <= $E_T$ < %.3f'%(rang[10], rang[11]), '%.3f <= $E_T$ < %.3f'%(rang[11], rang[12]),'%.3f <= $E_T$ < %.3f'%(rang[12], rang[13]),'%.3f <= $E_T$ < %.3f'%(rang[13], rang[14])]
        labs4 = ['%.3f <= $E_T$ < %.3f'%(rang[14], rang[15]), '%.3f <= $E_T$ < %.3f'%(rang[15], rang[16]), '%.3f <= $E_T$ < %.3f'%(rang[16], rang[17]),'%.3f <= $E_T$ < %.3f'%(rang[17], rang[18]),'%.3f <= $E_T$ < %.3f'%(rang[18], rang[19])]
        labs5 = ['%.3f <= $E_T$ < %.3f'%(rang[19], rang[20]), '%.3f <= $E_T$ < %.3f'%(rang[20], rang[21]), '%.3f <= $E_T$ < %.3f'%(rang[21], rang[22]),'%.3f <= $E_T$ < %.3f'%(rang[22], rang[23]),'%.3f <= $E_T$ < %.3f'%(rang[23], rang[24])]
        labs6 = ['%.3f <= $E_T$ < %.3f'%(rang[24], rang[25]), '%.3f <= $E_T$ < %.3f'%(rang[25], rang[26]), '%.3f <= $E_T$ < %.3f'%(rang[26], rang[27]),'%.3f <= $E_T$ < %.3f'%(rang[27], rang[28]),'%.3f <= $E_T$ < %.3f'%(rang[28], rang[29])]
  
    elif rangetype=='Pt':
        labs1 = ['$p_T$ < %.3f'%rang[0], '%.3f <= $p_T$ < %.3f'%(rang[0], rang[1]), '%.3f <= $p_T$ < %.3f'%(rang[1], rang[2]),'%.3f <= $p_T$ < %.3f'%(rang[2], rang[3]),'%.3f <= $p_T$ < %.3f'%(rang[3], rang[4])]
        labs2 = ['%.3f <= $p_T$ < %.3f'%(rang[4], rang[5]), '%.3f <= $p_T$ < %.3f'%(rang[5], rang[6]), '%.3f <= $p_T$ < %.3f'%(rang[6], rang[7]),'%.3f <= $p_T$ < %.3f'%(rang[7], rang[8]),'%.3f <= $p_T$ < %.3f'%(rang[8], rang[9])]
        labs3 = ['%.3f <= $p_T$ < %.3f'%(rang[9], rang[10]), '%.3f <= $p_T$ < %.3f'%(rang[10], rang[11]), '%.3f <= $p_T$ < %.3f'%(rang[11], rang[12]),'%.3f <= $p_T$ < %.3f'%(rang[12], rang[13]),'%.3f <= $p_T$ < %.3f'%(rang[13], rang[14])]
        labs4 = ['%.3f <= $p_T$ < %.3f'%(rang[14], rang[15]), '%.3f <= $p_T$ < %.3f'%(rang[15], rang[16]), '%.3f <= $p_T$ < %.3f'%(rang[16], rang[17]),'%.3f <= $p_T$ < %.3f'%(rang[17], rang[18]),'%.3f <= $p_T$ < %.3f'%(rang[18], rang[19])]
        labs5 = ['%.3f <= $p_T$ < %.3f'%(rang[19], rang[20]), '%.3f <= $p_T$ < %.3f'%(rang[20], rang[21]), '%.3f <= $p_T$ < %.3f'%(rang[21], rang[22]),'%.3f <= $p_T$ < %.3f'%(rang[22], rang[23]),'%.3f <= $p_T$ < %.3f'%(rang[23], rang[24])]
        labs6 = ['%.3f <= $p_T$ < %.3f'%(rang[24], rang[25]), '%.3f <= $p_T$ < %.3f'%(rang[25], rang[26]), '%.3f <= $p_T$ < %.3f'%(rang[26], rang[27]),'%.3f <= $p_T$ < %.3f'%(rang[27], rang[28]),'%.3f <= $p_T$ < %.3f'%(rang[28], rang[29])]
  
    elif rangetype=='E':
        labs1 = ['$E$ < %.3f'%rang[0], '%.3f <= $E$ < %.3f'%(rang[0], rang[1]), '%.3f <= $E$ < %.3f'%(rang[1], rang[2]),'%.3f <= $E$ < %.3f'%(rang[2], rang[3]),'%.3f <= $E$ < %.3f'%(rang[3], rang[4])]
        labs2 = ['%.3f <= $E$ < %.3f'%(rang[4], rang[5]), '%.3f <= $E$ < %.3f'%(rang[5], rang[6]), '%.3f <= $E$ < %.3f'%(rang[6], rang[7]),'%.3f <= $E$ < %.3f'%(rang[7], rang[8]),'%.3f <= $E$ < %.3f'%(rang[8], rang[9])]
        labs3 = ['%.3f <= $E$ < %.3f'%(rang[9], rang[10]), '%.3f <= $E$ < %.3f'%(rang[10], rang[11]), '%.3f <= $E$ < %.3f'%(rang[11], rang[12]),'%.3f <= $E$ < %.3f'%(rang[12], rang[13]),'%.3f <= $E$ < %.3f'%(rang[13], rang[14])]
        labs4 = ['%.3f <= $E$ < %.3f'%(rang[14], rang[15]), '%.3f <= $E$ < %.3f'%(rang[15], rang[16]), '%.3f <= $E$ < %.3f'%(rang[16], rang[17]),'%.3f <= $E$ < %.3f'%(rang[17], rang[18]),'%.3f <= $E$ < %.3f'%(rang[18], rang[19])]
        labs5 = ['%.3f <= $E$ < %.3f'%(rang[19], rang[20]), '%.3f <= $E$ < %.3f'%(rang[20], rang[21]), '%.3f <= $E$ < %.3f'%(rang[21], rang[22]),'%.3f <= $E$ < %.3f'%(rang[22], rang[23]),'%.3f <= $E$ < %.3f'%(rang[23], rang[24])]
        labs6 = ['%.3f <= $E$ < %.3f'%(rang[24], rang[25]), '%.3f <= $E$ < %.3f'%(rang[25], rang[26]), '%.3f <= $E$ < %.3f'%(rang[26], rang[27]),'%.3f <= $E$ < %.3f'%(rang[27], rang[28]),'%.3f <= $E$ < %.3f'%(rang[28], rang[29])]
  
    else:
        print('you broke it')
        
        
    col = ['xkcd:golden brown', 'xkcd:amber', 'xkcd:sandy brown', 'xkcd:grey brown', 'xkcd:pale peach']
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=[24,16])

    plothelper(ax1, labs1, fps[:5], tps[:5], auc[:5], col, fps[-1], tps[-1], auc[-1])
    plothelper(ax2, labs2, fps[5:10], tps[5:10], auc[5:10], col, fps[-1], tps[-1], auc[-1])
    plothelper(ax3, labs3, fps[10:15], tps[10:15], auc[10:15], col, fps[-1], tps[-1], auc[-1])
    plothelper(ax4, labs4, fps[15:20], tps[15:20], auc[15:20], col, fps[-1], tps[-1], auc[-1])
    plothelper(ax5, labs5, fps[20:25], tps[20:25], auc[20:25], col, fps[-1], tps[-1], auc[-1])
    plothelper(ax6, labs6, fps[25:30], tps[25:30], auc[25:30], col, fps[-1], tps[-1], auc[-1])
    plt.savefig('./'+rangetype+'_ranges_AUCs')
    
    
#make metrics for the no global model, these will be universal
def metrics_ng(X_test, selections):

    fps_ng = []
    tps_ng = []
    aucs_ng = []
    for selection in selections:
        preds = model_nog.predict(X_test[selection], batch_size=1000) 
        pfn_fp, pfn_tp, threshs = roc_curve(Y_test[selection][:,1], preds[:,1])

        fps_ng.append(pfn_fp)
        tps_ng.append(pfn_tp)

        # get area under the ROC curve
        auc = roc_auc_score(Y_test[selection][:,1], preds[:,1])
        aucs_ng.append(auc)
        print('PFN AUC:', auc)
    return fps_ng, tps_ng, aucs_ng