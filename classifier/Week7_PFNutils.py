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

data_path = '/fast_scratch/atlas_images/v01-45/' 

#this is the no global params model
model_nog = tf.keras.models.load_model(data_path+'w6_pfn_noglob.hdf5')

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

def metrics(model, X_te, X_globte, Y_test, select):
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
    ax2.set_ylabel('Rejection at 95\% efficiency')
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
def metrics_ng(model, X_test, Y_test, selections):

    fps_ng = []
    tps_ng = []
    aucs_ng = []
    preds_ng = []
    for selection in selections:
        preds = model.predict(X_test[selection], batch_size=1000) 
        pfn_fp, pfn_tp, threshs = roc_curve(Y_test[selection][:,1], preds[:,1])
        preds_ng.append(preds)

        fps_ng.append(pfn_fp)
        tps_ng.append(pfn_tp)

        # get area under the ROC curve
        auc = roc_auc_score(Y_test[selection][:,1], preds[:,1])
        aucs_ng.append(auc)
        print('PFN AUC:', auc)
    return fps_ng, tps_ng, aucs_ng, preds_ng

def model_noglob(X_train, X_val, X_test, Y_train, Y_val, Y_test, epochs, batch_size, filename):
    #run the model & create the metrics
    
    Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
   
    #make the model:
    pfn = PFN(input_dim=X_train.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)#, num_global_features =1)
    
    #try callbacks:
    callback = tf.keras.callbacks.ModelCheckpoint(data_path+filename, save_best_only=True)
    
    # train model
    history = pfn.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val), verbose=1, callbacks=[callback])
    
    return history

def ratio_plot(AUC, REG95, auc_nog, reg95_nog, eta_ranges):
    #first two are for model you want to examine, second two are for the no-glob model you want to compare it to(or use some other model as the baseline if you want, like global eta only)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12, 6])

    ax1.set_xlim(0., 3.1)
    ax1.set_ylabel('model AUC divided by no global features model AUC', fontsize=15)
    ax1.plot(eta_ranges, np.array(AUC)/np.array(auc_nog), c='xkcd:goldenrod', linewidth=2, marker='o', label='no ID, global $\eta$')
    ax1.set_xlabel('range in |$\eta$|')
    ax1.legend()

    ax2.set_xlim(0., 3.1)
    ax2.set_ylabel('model rejection at 95\% effeciency/ no-glob rejection', fontsize=15)
    ax2.plot(eta_ranges, np.array(REG95)/np.array(reg95_nog), c='xkcd:golden', linewidth=2, marker='o', label='no ID, global $\eta$')
    ax2.set_xlabel('range in |$\eta$|')
    ax2.legend()
    
#turn this into a util function:
def scorehist(preds, Y_test, eta_test):
    prob1, prob2 = preds.T
    lab1, lab2 = Y_test.T
    pip_mask = lab1 == 1 #note I'm not sure if I have these mixed up or not it might be the other way around lol
    pi0_mask = lab1 == 0
    pi0_pred = prob1[pi0_mask]
    pip_pred = prob1[pip_mask]
    
    #I still think I need both eta bins tho I'm sure there's a nicer way to do this
    eta_pi0 = eta_test[pi0_mask]
    eta_pip = eta_test[pip_mask]

    selec1 = abs(eta_pi0) < 0.5
    selec2 = (abs(eta_pi0) >= .5) & (abs(eta_pi0) < 1.)
    selec3 = (abs(eta_pi0) >= 1.) & (abs(eta_pi0) < 1.5)
    selec4 = (abs(eta_pi0) >= 1.5) & (abs(eta_pi0) < 2.)
    selec5 = (abs(eta_pi0) >= 2.) & (abs(eta_pi0) < 2.5)
    selec6 = (abs(eta_pi0) >= 2.5) & (abs(eta_pi0) < 3.1)

    eta_bins_0 = [selec1, selec2, selec3, selec4, selec5, selec6]

    selec1 = abs(eta_pip) < 0.5
    selec2 = (abs(eta_pip) >= .5) & (abs(eta_pip) < 1.)
    selec3 = (abs(eta_pip) >= 1.) & (abs(eta_pip) < 1.5)
    selec4 = (abs(eta_pip) >= 1.5) & (abs(eta_pip) < 2.)
    selec5 = (abs(eta_pip) >= 2.) & (abs(eta_pip) < 2.5)
    selec6 = (abs(eta_pip) >= 2.5) & (abs(eta_pip) < 3.1)

    eta_bins_p = [selec1, selec2, selec3, selec4, selec5, selec6]
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=[24,16])
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    ranges = ['|$\eta$| < .5', '.5< |$\eta$| < 1.', '1.< |$\eta$| < 1.5', '1.5< |$\eta$| < 2.', '2.< |$\eta$| < 2.5', '2.5< |$\eta$| < 3.']

    for i in range(len(eta_bins_0)):
        axes[i].set_xlim(0,1)
        axes[i].set_ylim(0,3e4)
        axes[i].set_title(ranges[i])
        axes[i].hist(pip_pred[eta_bins_p[i]], color='xkcd:light mustard', label='true pi+/-')
        axes[i].hist(pi0_pred[eta_bins_0[i]], color='xkcd:ochre', label=' true pi0')
        axes[i].legend()
        
        
def deltaR_plots(DR_ranges, predics, col):
    #plotting funciton for the below function; plot predicted probabilities as function of deltaR
    fig, ((ax1, ax2, ax3, ax7), (ax4, ax5, ax6, ax8)) = plt.subplots(2,4,figsize=[24,12])
    axes = [ax1, ax2, ax3, ax7, ax4, ax5, ax6, ax8]
    ranges = ['$\Delta R$ <' + str(np.round(DR_ranges[1], 4)), str(np.round(DR_ranges[1], 4))+ ' < $\Delta R$ < ' + str(np.round(DR_ranges[2], 4)), str(np.round(DR_ranges[2], 4)) + ' < $\Delta R$ < '+ str(np.round(DR_ranges[3], 4)), str(np.round(DR_ranges[3], 4))+' < $\Delta R$ < '+ str(np.round(DR_ranges[4], 4)), str(np.round(DR_ranges[4], 4))+' < $\Delta R$ < '+ str(np.round(DR_ranges[5], 4)), str(np.round(DR_ranges[5], 4))+' < $\Delta R$ < '+ str(np.round(DR_ranges[6], 4)), str(np.round(DR_ranges[6], 4))+' < $\Delta R$ < '+ str(np.round(DR_ranges[7],4)), str(np.round(DR_ranges[7], 4))+' < $\Delta R$ < '+ str(np.round(DR_ranges[8], 4))]

    for i in range(len(axes)):
        prob1, prob2 = predics[i].T
        axes[i].set_xlim(0,1)
        #axes[i].set_ylim(0,3e4)
        axes[i].set_title(ranges[i])
        axes[i].hist(prob1, color=col)
        

def deltaR_responseplots(file, model, col):
    #guess I'm just adding everything here, so here's the function to make a plot of the predictions of testing a (no-glob) model as a funciton of deltaR . Just give it the file with the data, the model to use, and a colour to plot
    X_test = file['arr_2']
    eta_test = file['arr_5']
    delR_test = file['arr_17']

    DR_ranges = np.linspace(0., max(delR_test)+.1, 9) #includes stop

    DR_sel = [abs(delR_test) < DR_ranges[1]]
    for i in range(1, len(DR_ranges)):
        selec_ = (abs(delR_test) >= DR_ranges[i-1]) & (abs(delR_test) < DR_ranges[i])
        DR_sel.append(selec_)

    predics = []
    for selection in DR_sel:
        preds = model_nog.predict(X_test[selection], batch_size=1000)
        predics.append(preds)
        
    deltaR_plots(DR_ranges, predics, col=col)
    
def datatofile(dat, outfile, size):
    X, clus_eta, clus_pt, clus_E, clus_et, deltar = dat
    X_all = np.array(X[:size])
    eta_all = np.array(clus_eta[:size])
    pt_all = np.array(clus_pt[:size])
    E_all = np.array(clus_E[:size])
    et_all = np.array(clus_et[:size])
    deltar_all = np.array(deltar[:size])

    (X_train, X_val, X_test,  
     eta_train, eta_val, eta_test, 
     ET_train, ET_val, ET_test, 
     pt_train, pt_val, pt_test, 
     Eng_train, Eng_val, Eng_test,
     deltar_train, deltar_val, deltar_test) = data_split(X_all, eta_all, et_all, pt_all, E_all, deltar_all, val=100, test=int(size/2.))

    np.savez(data_path+outfile, X_train, X_val, X_test, eta_train, eta_val, eta_test, ET_train, ET_val, ET_test, pt_train, pt_val, pt_test, Eng_train, Eng_val, Eng_test, deltar_train, deltar_val, deltar_test)