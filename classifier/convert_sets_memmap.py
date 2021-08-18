#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import os
import Week7PFNutils as W
import graph_util_splitdelta as gu

#====================
# Functions =========
#====================

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2) # ~gu~

def find_max_dim(events, event_dict):
    ''' This function is designed to return the sizes of a numpy array such that we are efficient
    with zero padding. 
    Inputs:
        events: filtered list of events to choose from in an Nx3 format for event, track, cluster index 
        event_tree: the event tree dictionary
    Returns:
        3-tuple consisting of (number of events, maximum cluster_size, 4), 4 because of how we have structured
        the X data format in energyFlow to be Energy, Eta, Phi, sampling_layer
    '''
    nEvents = len(events) 
    max_clust = 0
    for i in range(nEvents):
        evt = events[i,0]
        clust_idx = events[i,2]
        num_in_clust = len(event_dict['cluster_cell_ID'][evt][clust_idx])
        if num_in_clust > max_clust:
            max_clust = num_in_clust
            

    return (nEvents, max_clust+4, 4) #change to 4 ~gu~

def find_max_dim_tuple(events, event_dict):
    nEvents = len(events)
    max_clust = 0
    
    
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 6 #will this have to change? **
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust
        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 4 for energy, eta, phi, sample layer
    return (nEvents, max_clust, 4) #6->4 ~gu~

def dict_from_tree(tree, branches=None, np_branches=None):
    ''' Loads branches as default awkward arrays and np_branches as numpy arrays. '''
    dictionary = dict()
    if branches is not None:
        for key in branches:
            branch = tree.arrays()[key]#############
            dictionary[key] = branch
            
    if np_branches is not None:
        for np_key in np_branches:
            np_branch = np.ndarray.flatten(tree.arrays()[np_key].to_numpy()) #########
            dictionary[np_key] = np_branch
    
    if branches is None and np_branches is None:
        raise ValueError("No branches passed to function.")
        
    return dictionary # ~gu~

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec #~gu~ could just use append but okay

#====================
# Metadata ==========
#====================
track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_hitsTruthE',
                     'truthPartEta', 'truthPartPhi', 'truthPartPt', 'truthPartMass','cluster_Pt', "nTruthPart", 
                     "truthPartPdgId", "cluster_Eta", "cluster_Phi", "truthPartE", "cluster_ENG_CALIB_TOT", "cluster_E"]

np_event_branches = ["mcChannelNumber","nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]

data_path = '/fast_scratch/atlas_images/v01-45/'

#====================
# File setup ========
#====================

def get_files(datatype, Nfile, IDnum):
    #give it datatype i.e. /rho/, /pi0/, /pipm/
    fileNames = []
    
    file_prefix_pipm = data_path + datatype + 'user.angerami.' + IDnum + '.OutputStream._000' #001.root
    for i in range(1,Nfile+1):
        endstring = f'{i:03}'
        fileNames.append(file_prefix_pipm + endstring + '.root')
    return fileNames



#====================
# Load Data Files ===
#====================

def RUN(fileNames, outname, Labels, Nfile, isRho=False, mask=None):
    #you kinda need a different case if its rho... lets build the mask-making and the closest cluster stuff all in here!
                   
    ## GEOMETRY DICTIONARY ##
    geo_file = ur.open(data_path+'cell_geo.root')
    CellGeo_tree = geo_file["CellGeo"]
    geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

    # cell geometry data
    cell_geo_ID = geo_dict['cell_geo_ID']
    cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

    # Use this to compare with the dimensionality of new events
    firstArray = True

    ## MEMORY MAPPED ARRAY ALLOCATION ##
    X_large = np.lib.format.open_memmap('/data/atlas/Kaela/X_large_K.npy', mode='w+', dtype=np.float64, #changed these to be mine
                           shape=(1700000,1500,4), fortran_order=False, version=None)
    Y_large = np.lib.format.open_memmap('/data/atlas/Kaela/Y_large_K.npy', mode='w+', dtype=np.float64,
                           shape=(1700000,1), fortran_order=False, version=None)


    k = 1 # tally used to keep track of file number
    tot_nEvts = 0 # used for keeping track of total number of events
    max_nPoints = 0 # used for keeping track of the largest 'point cloud'
    t_tot = 0 # total time

    for j, currFile in enumerate(fileNames):
        # Check for file, a few are missing
        if not os.path.isfile(currFile):
            print()
            print('File '+currFile+' not found..')
            print()
            k += 1
            continue

        else:
            print()
            print('Working on File: '+str(currFile)+' - '+str(k)+'/'+str(Nfile))
            k += 1

        t0 = t.time()
        ## EVENT DICTIONARY ##
        event = ur.open(currFile)
        event_tree = event["EventTree"]
        event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)

        Label = Labels[j]
        
        #===================
        # APPLY CUTS =======
        #===================
        # create ordered list of events to use for index slicing
        nEvents = len(event_dict['eventNumber'])
        all_events = np.arange(0,nEvents,1,dtype=np.int32)

        nCluster = event_dict['nCluster']
        filtered_event_mask = nCluster != 0 #just remove empty events, ~gu~
        
        truthID = event_dict['truthPartPdgId']
        
        filtered_event = all_events[filtered_event_mask]
        
        #just add this to the passed mask?
        if isRho:
            mask211 = makemask(truthID, 211)
            maskn211 = makemask(truthID, -211)
            
            #then need to I guess..... just..... do this the long way:(
            truthE = event_dict['cluster_hitsTruthE']
            trutheta = event_dict['truthPartEta']
            truthphi = event_dict['truthPartPhi']
            truthpt = event_dict['truthPartPt']
            truthmass = event_dict['truthPartMass']
            clusE = event_dict['cluster_E']
            clusEta = event_dict['cluster_Eta']
            clusPhi = event_dict['cluster_Phi']
            clusPt = event_dict['cluster_Pt']
            
            #to save space I'll just reuse these variable names? after?
            truthE_sel = truthE[mask211]
            truthID_sel = truthID[mask211]
            trutheta_sel = trutheta[mask211]
            truthphi_sel = truthphi[mask211]
            truthpt_sel = truthpt[mask211]
            truthmass_sel = truthmass[mask211]
            clusE_sel = clusE[mask211]
            clusEta_sel = clusEta[mask211]
            clusPhi_sel = clusPhi[mask211]
            clusPt_sel = clusPt[mask211]
            clusmass_sel = []
            for i in range(len(clusE_sel)): #define cluster mass to be zero :)
                clusmass_sel.append(np.zeros(len(clusE_sel[i])))
                
            #replace all the vars with those of the closest cluter only
            truID1, trueta1, truphi1, trupt1, trum1, clus_e1, clus_eta1, clus_phi1, clus_pt1, clus_mass1, full_indices1, mindist_inds1 = gu.find_closest2(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel, clusPt_sel, clusmass_sel, partID1, partID2)
            
            #to save space I'll just reuse these variable names now.....
            truthE_sel = truthE[maskn211]
            truthID_sel = truthID[maskn211]
            trutheta_sel = trutheta[maskn211]
            truthphi_sel = truthphi[maskn211]
            truthpt_sel = truthpt[maskn211]
            truthmass_sel = truthmass[maskn211]
            clusE_sel = clusE[maskn211]
            clusEta_sel = clusEta[maskn211]
            clusPhi_sel = clusPhi[maskn211]
            clusPt_sel = clusPt[maskn211]
            clusmass_sel = []
            for i in range(len(clusE_sel)): #define cluster mass to be zero :)
                clusmass_sel.append(np.zeros(len(clusE_sel[i])))
                
            #replace all the vars with those of the closest cluter only
            truID2, trueta2, truphi2, trupt2, trum2, clus_e2, clus_eta2, clus_phi2, clus_pt2, clus_mass2, full_indices2, mindist_inds2 = gu.find_closest2(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel, clusPt_sel, clusmass_sel, partID1, partID2)
            
        
        #here: provision for if rho? I think the mask should be built in instead, why not right?
        print("* Selected events with clusters: ", len(filtered_event), "/", len(all_events))

        t1 = t.time()
        events_cuts_time = t1 - t0

        #============================================#
        ## CREATE INDEX ARRAY FOR  CLUSTERS ##
        #============================================#
        event_indices = []
        t0 = t.time()

        for evt in filtered_event:
            # pull cluster number, don't need zero index as it's loaded as a np array
            nClust = event_dict["nCluster"][evt]
            cluster_idx = np.arange(nClust)

            ## Cluster properties
            clusterEs = event_dict["cluster_E"][evt].to_numpy()
            clus_phi = event_dict["cluster_Phi"][evt].to_numpy()
            clus_eta = event_dict["cluster_Eta"][evt].to_numpy()

            ## ENERGY SELECTION AND ETA SELECTION
            e_mask = clusterEs > 0.5
            selection =  e_mask  
            selected_clusters = cluster_idx[selection]

            ## CREATE LIST ##
            if np.count_nonzero(selection) > 0:
                event_indices.append((evt, 0, selected_clusters))
                if (evt % 10000==0):
                    print("** Event ", evt, " has ", np.count_nonzero(selection), "/" ,nClust," selected clusters")

        event_indices = np.array(event_indices, dtype=np.object_)
        t1 = t.time()
        indices_time = t1 - t0 #~gu~

        #=========================#
        ## DIMENSIONS OF X ARRAY ##
        #=========================#
        t0 = t.time()
        max_dims = find_max_dim_tuple(event_indices, event_dict)
        evt_tot = max_dims[0]
        tot_nEvts += max_dims[0]
        # keep track of the largest point cloud to use for saving later
        if max_dims[1] > max_nPoints:
            max_nPoints = max_dims[1]


        print('* Events with selected clusters: '+str(evt_tot))
        print('* Total number of cells: '+str(max_dims[1]))
        print('* Dim of largest point cloud: '+str(max_nPoints))


        # Create arrays
        Y_new = np.ones((max_dims[0],1)) * Label
        X_new = np.zeros(max_dims)
        t1 = t.time()
        find_create_max_dims_time = t1 - t0    


        #===================#
        ## FILL IN ENTRIES ##==============================================================
        #===================#
        t0 = t.time()
        for i in range(max_dims[0]):
            # pull all relevant indices
            evt = event_indices[i,0]
            track_idx = event_indices[i,1]
            # recall this now returns an array
            cluster_nums = event_indices[i,2]

            ##############
            ## CLUSTERS ##
            ##############
            # set up to have no clusters, further this with setting up the same thing for tracks
            target_ENG_CALIB_TOT = -1
            if cluster_nums is not None: # and Label>=0:

                # find averaged center of clusters
                cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()
                cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()
                av_Eta = np.mean(cluster_Eta)
                av_Phi = np.mean(cluster_Phi)

                nClust_current_total = 0
                target_ENG_CALIB_TOT = 0

                for c in cluster_nums:            
                    # cluster data
                    target_ENG_CALIB_TOT += event_dict['cluster_ENG_CALIB_TOT'][evt][c]
                    cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
                    nInClust = len(cluster_cell_ID)
                    cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()            
                    cell_indices = find_index_1D(cluster_cell_ID, cell_ID_dict)

                    cluster_cell_Eta = geo_dict['cell_geo_eta'][cell_indices]
                    cluster_cell_Phi = geo_dict['cell_geo_phi'][cell_indices]
                    cluster_cell_sampling = geo_dict['cell_geo_sampling'][cell_indices]

                    # input all the data
                    # note here we leave the fourth entry zeros (zero for flag!!!)
                    low = nClust_current_total
                    high = low + nInClust
                    X_new[i,low:high,0] = np.log(cluster_cell_E)
                    # Normalize to average cluster centers (or to parent cluster center)
                    X_new[i,low:high,1] = cluster_cell_Eta - av_Eta #cluster_cell_Eta - event_dict['cluster_Eta'][evt][c]
                    X_new[i,low:high,2] = cluster_cell_Phi - av_Phi #cluster_cell_Phi -event_dict['cluster_Phi'][evt][c]
                    X_new[i,low:high,3] = cluster_cell_sampling * 0.1

                    nClust_current_total += nInClust #~gu~


                ###########################
                ## Classification labels ##
                ###########################
                #if Label>=0:
                #Y_new[i] = Label

        #####################################################
        t1 = t.time()
        array_construction_time = t1 - t0


        #=======================#
        ## ARRAY CONCATENATION ##
        #=======================#
        t0 = t.time()
        # Write to X
        old_tot = tot_nEvts - max_dims[0]
        X_large[old_tot:tot_nEvts, :max_dims[1], :4] = np.ndarray.copy(X_new) #change these 6s too?
        # pad the remainder with zeros (just to be sure)
        fill_shape = (tot_nEvts - old_tot, 1500 - max_dims[1], 4)
        X_large[old_tot:tot_nEvts, max_dims[1]:1500, :] = np.zeros(fill_shape)

        # Write to Y
        Y_large[old_tot:tot_nEvts,:] = np.ndarray.copy(Y_new)

        t1 = t.time()
        time_to_memmap = t1-t0
        thisfile_t_tot = events_cuts_time+find_create_max_dims_time+indices_time\
              +array_construction_time+time_to_memmap
        t_tot += thisfile_t_tot




        ##########################33

        print('Array dimension: '+str(max_dims))
        print('Time to create dicts and select events: '+str(events_cuts_time))
        print('Time to construct index array: '+str(indices_time))
        print('Time to find dimensions and make new array: '+str(find_create_max_dims_time))
        print('Time to populate elements: '+str(array_construction_time))

        print('Time to copy to memory map: '+str(time_to_memmap))
        print('Time for this file: '+str(thisfile_t_tot))
        print('Total events: '+str(tot_nEvts))
        print('Current size: '+str((tot_nEvts,max_nPoints,4)))
        print('Total time: '+str(t_tot))
        print()

    t0 = t.time()
    X = np.lib.format.open_memmap('/data/atlas/Kaela/X_'+str(Nfile)+'_files.npy', #making my own version of these?
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 4))
    np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
    del X_large
    os.system('rm /data/atlas/Kaela/X_large_K.npy')

    Y = np.lib.format.open_memmap('/data/atlas/Kaela/Y_'+str(Nfile)+'_files.npy',
                                 mode='w+', dtype=np.float64, shape=(tot_nEvts,1))
    np.copyto(dst=Y, src=Y_large[:tot_nEvts,:], casting='same_kind', where=True)
    del Y_large
    os.system('rm /data/atlas/Kaela/Y_large_K.npy')

    t1 = t.time()
    print()
    print('Time to copy new and delete old: '+str(t1-t0)+' (s)')
    print()
    
    
    
def doit(filetypes, mask, Nfiles, outname):
    filenames = []
    labels = []
    for i in range(len(filetypes)):
        
        if filetypes[i] == 'pi0/':
            IDnum = '24559740'
            x = get_files(filetypes[i], Nfiles, IDnum)
            filenames.append(x)
            labels.append(np.zeros(len(x)))
            
        elif filetypes[i] == 'pipm/':
            IDnum = '24559744'
            x = get_files(filetypes[i], Nfiles, IDnum)
            filenames.append(x)
            labels.append(np.ones(len(x)))
            
        elif filetypes[i] == 'rho/':
            IDnum = '24409109'
            isRho = True
            x = get_files(filetypes[i], Nfiles, IDnum)
            filenames.append(x)
            labels.append(np.ones(len(x))*2) #& add stuff here about how to do the closest cluster thing - will have to add that in in the next step 
        else:
            raise ValueError(filetypes[i]+'Not a valid filetype')         
    
    filenames = [item for sublist in filenames for item in sublist]
    Nfiles = len(filenames)
    
    RUN(filenames, outname, labels, Nfiles, isRho, mask) #so now there's the same number of labels as there are files???