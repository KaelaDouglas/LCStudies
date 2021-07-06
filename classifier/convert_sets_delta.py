import uproot as ur
import awkward as ak
import numpy as np
from glob import glob

from tensorflow.keras.utils import to_categorical

import sys
sys.path.append('/Users/swiatlow/Code/ML4P/LCStudies')
sys.path.append('/home/mswiatlowski/start_tf/LCStudies')
import graph_util as gu

#data_path = '/Users/swiatlow/Data/caloml/graph_data/'
data_path = '/fast_scratch/atlas_images/v01-45/'

rho_list = [data_path+ 'delta_medium.root']

def convertFile(filename):
    print('Working on {}'.format(filename))
    
    tree = ur.open(filename)['EventTree']
    geotree = ur.open(filename)['CellGeo']

    geo_dict = gu.loadGraphDictionary(geotree)

    print('Loading data')
    # should I remove things over 2000?

    ## First, load all information we want
    cell_id = gu.loadArrayBranchFlat('cluster_cell_ID', tree, 2000)
    cell_e  = gu.loadArrayBranchFlat('cluster_cell_E', tree, 2000)

    cell_eta  = gu.convertIDToGeo(cell_id, 'cell_geo_eta', geo_dict)
    cell_phi  = gu.convertIDToGeo(cell_id, 'cell_geo_phi', geo_dict)
    cell_samp = gu.convertIDToGeo(cell_id, 'cell_geo_sampling', geo_dict)

    clus_phi = gu.loadVectorBranchFlat('cluster_Phi', tree)
    clus_eta = gu.loadVectorBranchFlat('cluster_Eta', tree)
    clus_pt = gu.loadVectorBranchFlat('cluster_Pt', tree)

    clus_e   = gu.loadVectorBranchFlat('cluster_E', tree)
    
    clus_e_t = clus_e / np.cosh(np.array(clus_eta))

    clus_targetE = gu.loadVectorBranchFlat('cluster_ENG_CALIB_TOT', tree)

    ## Now, setup selections
    #eta_mask = (abs(clus_eta) < 0.7) | (abs(clus_eta) > .6)
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
    outname = filename.replace('root', 'npz')
    np.savez(outname, X=X, clus_eta=clus_eta[selection], clus_e=clus_e[selection], clus_pt=clus_pt[selection], clus_e_t=clus_e_t[selection])
    print('Done! {}'.format(outname))

for rho_file in rho_list:
    convertFile(rho_file)



