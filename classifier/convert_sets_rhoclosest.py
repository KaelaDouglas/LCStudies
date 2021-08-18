import uproot as ur
import awkward as ak
import numpy as np
from glob import glob

from tensorflow.keras.utils import to_categorical

import sys
sys.path.append('/Users/swiatlow/Code/ML4P/LCStudies')
sys.path.append('/home/mswiatlowski/start_tf/LCStudies')
import graph_util_splitdelta as gu

#data_path = '/Users/swiatlow/Data/caloml/graph_data/'
data_path = '/fast_scratch/atlas_images/v01-45/'

rho_list = [data_path+ 'rho_medium.root']

def convertFile(filename, mask, outname):
    #this is the regular one, should still work for any of them.
    print('Working on {}'.format(filename))
    
    tree = ur.open(filename)['EventTree']
    geotree = ur.open(filename)['CellGeo']

    geo_dict = gu.loadGraphDictionary(geotree)

    print('Loading data')
    # should I remove things over 2000?
    
    #need notflat E for deltaR here -note this isn't needed for closest cluster below
    notflatE = tree.arrays('cluster_E').cluster_E[mask]
    
    #all the stuff for deltaR:
    truID = gu.loadVectorBranchFlat('truthPartPdgId', tree, mask)
    trueta = gu.loadVectorBranchFlat('truthPartEta', tree, mask)
    truphi = gu.loadVectorBranchFlat('truthPartPhi', tree, mask)
    trupt = gu.loadVectorBranchFlat('truthPartPt', tree, mask)
    trum = gu.loadVectorBranchFlat('truthPartMass', tree, mask)
    
    deltaR = gu.make_deltaR(truID, trueta, truphi, trupt, trum, notflatE, partID1, partID2)
    
    ## First, load all information we want
    cell_id = gu.loadArrayBranchFlat('cluster_cell_ID', tree, 2000, deltamask)
    cell_e  = gu.loadArrayBranchFlat('cluster_cell_E', tree, 2000, deltamask)

    cell_eta  = gu.convertIDToGeo(cell_id, 'cell_geo_eta', geo_dict)
    cell_phi  = gu.convertIDToGeo(cell_id, 'cell_geo_phi', geo_dict)
    cell_samp = gu.convertIDToGeo(cell_id, 'cell_geo_sampling', geo_dict)
    
    clus_phi = gu.loadVectorBranchFlat('cluster_Phi', tree, deltamask)
    clus_eta = gu.loadVectorBranchFlat('cluster_Eta', tree, deltamask)
    clus_pt = gu.loadVectorBranchFlat('cluster_Pt', tree, deltamask)
    
    clus_e   = gu.loadVectorBranchFlat('cluster_E', tree, deltamask) #it is sooo not helpful that gu flattens it right off the bat
    
    clus_e_t = clus_e / np.cosh(np.array(clus_eta))
    
    ## Now, setup selections
    e_mask = clus_e > 0.5

    selection = e_mask # & eta_mask
    
    ## Now, normalize
    print('Normalizing')    
    # normalize cell location relative to cluster center
    cell_eta = np.nan_to_num(cell_eta - clus_eta[:, None])
    cell_phi = np.nan_to_num(cell_phi - clus_phi[:, None])
    #normalize energy by taking log
    cell_e = np.nan_to_num(np.log(cell_e), posinf = 0, neginf=0)
    #normalize sampling by 0.1
    cell_samp = cell_samp * 0.1

    print('Writing out')
    #prepare outputs
    X = np.stack((cell_e[selection],
                    cell_eta[selection],
                    cell_phi[selection],
                    cell_samp[selection]),
                    axis = 2)

    
    #Now we save. prepare output filename.
    #outname = filename.replace('root', 'npz')
    np.savez(outname, X=X, clus_eta=clus_eta[selection], clus_e=clus_e[selection], clus_pt=clus_pt[selection], clus_e_t=clus_e_t[selection], deltaR = np.array(deltaR[selection])) #maybe?? mayeb????? it doesn't matter if I don't use the selection???
    print('Done! {}'.format(outname))


def convertFile_closestonly(filename, deltamask, outname, partID1, partID2):
    #partID1 should be the particle of interest
    print('Working on {}'.format(filename))
    
    tree = ur.open(filename)['EventTree']
    
    geotree = ur.open(filename)['CellGeo']

    geo_dict = gu.loadGraphDictionary(geotree)
    
    brnch = ['cluster_hitsTruthE', 'cluster_hitsTruthIndex', 'truthPartPdgId', 'truthPartEta', 'truthPartPhi', 
         'truthPartPt', 'truthPartMass', 'cluster_E', 'cluster_Eta', 'cluster_Phi','cluster_Pt']
    
    print('Loading data')
    
    #need to load up not-flat stuff for finding the closest clusters
    branches = tree.arrays(expressions=brnch)
    
    truthE = branches.cluster_hitsTruthE
    truthIndex = branches.cluster_hitsTruthIndex
    truthID = branches.truthPartPdgId
    trutheta = branches.truthPartEta
    truthphi = branches.truthPartPhi
    truthpt = branches.truthPartPt
    truthmass = branches.truthPartMass
    clusE = branches.cluster_E
    clusEta = branches.cluster_Eta
    clusPhi = branches.cluster_Phi
    clusPt = branches.cluster_Pt

    truthE_sel = truthE[deltamask]
    truthIndex_sel = truthIndex[deltamask]
    truthID_sel = truthID[deltamask]
    trutheta_sel = trutheta[deltamask]
    truthphi_sel = truthphi[deltamask]
    truthpt_sel = truthpt[deltamask]
    truthmass_sel = truthmass[deltamask]
    clusE_sel = clusE[deltamask]
    clusEta_sel = clusEta[deltamask]
    clusPhi_sel = clusPhi[deltamask]
    clusPt_sel = clusPt[deltamask]
    clusmass_sel = []
    for i in range(len(clusE_sel)): #define cluster mass to be zero :)
        clusmass_sel.append(np.zeros(len(clusE_sel[i])))
      
    #replace all the vars with those of the closest cluter only
    truID, trueta, truphi, trupt, trum, clus_e, clus_eta, clus_phi, clus_pt, clus_mass, full_indices, mindist_inds = gu.find_closest2(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel, clusPt_sel, clusmass_sel, partID1, partID2)
     
    #pass it the mindist indices to make it the right shape!
    cell_id = gu.loadArrayBranchFlat('cluster_cell_ID', tree, 2000, deltamask,empty=full_indices,mindist_inds=mindist_inds)
    cell_e = gu.loadArrayBranchFlat('cluster_cell_E', tree, 2000, deltamask, empty=full_indices, mindist_inds=mindist_inds) 
    
    #correct mindist shape carries over via cell_id
    cell_eta = gu.convertIDToGeo(cell_id, 'cell_geo_eta', geo_dict) 
    cell_phi = gu.convertIDToGeo(cell_id, 'cell_geo_phi', geo_dict)
    cell_samp = gu.convertIDToGeo(cell_id, 'cell_geo_sampling', geo_dict)
   
    
    #all the stuff for deltaR:
    truID = ak.flatten(truID).to_numpy()
    trueta = ak.flatten(trueta).to_numpy()
    truphi = ak.flatten(truphi).to_numpy()
    trupt = ak.flatten(trupt).to_numpy()
    trum = ak.flatten(trum).to_numpy()
    
    deltaR = gu.make_deltaR(truID, trueta, truphi, trupt, trum, clus_e, partID1, partID2, singlecluster=True) 
    #because its only the closest cluster, all the cluster stuff is already flat - i.e. there's only one cluster being examined so only one element at each index of the cluster arrays. so replacing *** notflatE *** with simple flat clus_e here should get deltaR to the right size.   

    
    clus_phi = clus_phi.to_numpy()
    clus_eta = clus_eta.to_numpy()
    clus_pt = clus_pt.to_numpy()
    
    clus_e = clus_e.to_numpy() 
    clus_e_t = clus_e / np.cosh(np.array(clus_eta))

    e_mask = clus_e > 0.5

    selection = e_mask 
    
    print('Normalizing')    
    # normalize cell location relative to cluster center
    cell_eta = np.nan_to_num(cell_eta - clus_eta[:, None])
    cell_phi = np.nan_to_num(cell_phi - clus_phi[:, None])
    #normalize energy by taking log
    cell_e = np.nan_to_num(np.log(cell_e), posinf = 0, neginf=0)
    #normalize sampling by 0.1
    cell_samp = cell_samp * 0.1

    print('Writing out')
    #prepare outputs
    X = np.stack((cell_e[selection],
                    cell_eta[selection],
                    cell_phi[selection],
                    cell_samp[selection]),
                    axis = 2)

    
    np.savez(outname, X=X, clus_eta=clus_eta[selection], clus_e=clus_e[selection], clus_pt=clus_pt[selection], clus_e_t=clus_e_t[selection], deltaR = np.array(deltaR[selection])) 
    print('Done! {}'.format(outname))


def doit(mask, outname, partID1, partID2):
    for del_file in del_list:
        convertFile(del_file, mask, outname, partID1, partID2)

def doit_cc(mask, outname, partID1, partID2):
    for del_file in del_list:
        convertFile_closestonly(del_file, mask, outname, partID1, partID2)



