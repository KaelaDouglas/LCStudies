# python
import numpy as np

# physics
import uproot as ur
import awkward as ak
import vector as vec

# visualization tools
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize 

data_path = '/fast_scratch/atlas_images/v01-45/'

#branch info:
track_branches = ['trackEta_EMB1', 'trackPhi_EMB1', 'trackEta_EMB2', 'trackPhi_EMB2', 'trackEta_EMB3', 'trackPhi_EMB3',
                  'trackEta_TileBar0', 'trackPhi_TileBar0', 'trackEta_TileBar1', 'trackPhi_TileBar1',
                  'trackEta_TileBar2', 'trackPhi_TileBar2']

event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]


#helper functions:

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2) 

## Notes, change this to accept special branches to convert to numpy arrays!!
def dict_from_event_tree(_event_tree, _branches):
    ''' The purpose for this separate function is to load np arrays where possible. '''
    _special_keys = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]
    _dict = dict()
    for _key in _branches:
        if _key in _special_keys:
            _branch = _event_tree.arrays(filter_name=_key)[_key].to_numpy()
        else:
            _branch = _event_tree.arrays(filter_name=_key)[_key]
        _dict[_key] = _branch
    return _dict

def dict_from_tree_branches(_tree, _branches):
    ''' Helper function to put event data in branches to make things easier to pass to functions,
    pretty self explanatory. '''
    _dict = dict()
    for _key in _branches:
        _branch = _tree.arrays(filter_name=_key)[_key]
        _dict[_key] = _branch
    return _dict

def dict_from_tree_branches_np(_tree, _branches):
    ''' Helper function to put event data in branches to make things easier to pass to functions,
    pretty self explanatory. This always returns np arrays in the dict. '''
    _dict = dict()
    for _key in _branches:
        _branch = np.ndarray.flatten(_tree.arrays(filter_name=_key)[_key].to_numpy())
        _dict[_key] = _branch
    return _dict

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. _values are the IDs to search for. _dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec

def to_xyz(_coords):
    ''' Simple geometric conversion to xyz from eta, phi, rperp (READ: in this order)
    There is an elegant way to generalize this to be flexible for 1d or 2d, for now 2d
    Inputs: np array of shape (N, 3) where columns are [eta, phi, rPerp]
    Outputs: np array of shape (N, 3) where columns are [x,y,z] '''
    _eta = _coords[:,0]
    _phi = _coords[:,1]
    _rperp = _coords[:,2]
    _theta = 2*np.arctan( np.exp(-_eta) )
    
    cell_x = _rperp*np.cos(_phi)
    cell_y = _rperp*np.sin(_phi)
    cell_z = _rperp/np.tan(_theta)
    
    return np.column_stack([cell_x,cell_y,cell_z])

def plottracks(cluster_xyz, tracks_xyz): #more useful like this I'd say
    colors = ['xkcd:burnt yellow', 'xkcd:indian red', 'xkcd:greyish', 'xkcd:mud brown', 'xkcd:tomato', 'xkcd:brownish pink', 'xkcd:pumpkin orange', 'xkcd:peachy pink', 'xkcd:dried blood'] #these are just for different clusters, need more than this!

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=30, azim=65)
    for i in range(len(cluster_xyz)):
        ax.scatter(cluster_xyz[i][:,0], cluster_xyz[i][:,1], cluster_xyz[i][:,2], color=colors[i],
                  s=20)
    for i in range(len(tracks_xyz)):
        ax.plot(tracks_xyz[i][:,0], tracks_xyz[i][:,1], tracks_xyz[i][:,2], color='black')
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_zlabel('z', fontsize=14, fontweight='bold')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=10, azim=105)
    for i in range(len(cluster_xyz)):
        ax.scatter(cluster_xyz[i][:,0], cluster_xyz[i][:,1], cluster_xyz[i][:,2], color=colors[i],
                  s=20)
    for i in range(len(tracks_xyz)):
        ax.plot(tracks_xyz[i][:,0], tracks_xyz[i][:,1], tracks_xyz[i][:,2], color='black')

    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_zlabel('z', fontsize=14, fontweight='bold')
    

def loadup(filename):
    my_event = ur.open(data_path+filename)
    event_tree = my_event['EventTree']
    event_dict = dict_from_event_tree(event_tree, event_branches)
    track_dict = dict_from_tree_branches(event_tree, track_branches)
    
    #& geometry separately:
    geo_file = ur.open(data_path+'cell_geo.root')
    CellGeo_tree = geo_file["CellGeo"]
    geo_dict = dict_from_tree_branches_np(CellGeo_tree, geo_branches)
    idx_dict = dict( zip( geo_dict['cell_geo_ID'], np.arange(len(geo_dict['cell_geo_ID'])) ) )
    
    return event_dict, geo_dict, track_dict, idx_dict
    
def run(event_dict, geo_dict, track_dict, idx_dict):
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)
    print('All events, shape '+str(all_events.shape))
    
    #make cuts to the events:

    # SINGLE TRACK CUT
    single_track_mask = event_dict['nTrack'] == np.full(nEvents, 1)
    filtered_event = all_events[single_track_mask]
    print('Single Track Indices, shape '+str(np.shape(filtered_event)))

    # CENTRAL TRACK CUT
    trackEta_EMB1 = ak.flatten(track_dict['trackEta_EMB1'][filtered_event]).to_numpy()
    central_track_mask = np.abs(trackEta_EMB1) < .7
    filtered_event = filtered_event[central_track_mask]
    # Save this for processing track only files later
    central_single_tracks = filtered_event
    print('Central track indices, shape '+str(np.shape(filtered_event)))
   
    #match clusters with tracks
    clusters_tracks = []

    for _evt in filtered_event:
        # pull cluster number, don't need zero as it's loaded as a np array
        _nClust = event_dict["nCluster"][_evt]

        ## DELTA R ##
        if _nClust != 0:
            # pull coordinates of tracks and clusters from event
            # we can get away with the zeroth index because we are working with single track events
            _trackCoords = np.array([event_dict["trackEta"][_evt][0],
                                     event_dict["trackPhi"][_evt][0]])
            _clusterCoords = np.stack((event_dict["cluster_Eta"][_evt].to_numpy(),
                                       event_dict["cluster_Phi"][_evt].to_numpy()), axis=1)

            _DeltaR = DeltaR(_clusterCoords, _trackCoords)
            _DeltaR_mask = _DeltaR < .2

            if np.any(_DeltaR_mask):
                clusters_tracks.append(_evt)

    print('Central tracks with matched clusters, shape '+str(np.shape(clusters_tracks)))
    print('Options: ', clusters_tracks[:30])
    
    event = int(input('Choose one of the above:'))
    
    ak_cluster_cell_ID = event_dict['cluster_cell_ID'][event]
    cell_geo_ID = geo_dict['cell_geo_ID']
    nClust = len(ak_cluster_cell_ID)

    clusters = []
    for j in range(nClust):

        # find cluster size
        _nInClust = len(ak_cluster_cell_ID[j])

        # make empty array of cluster info
        _cluster = np.zeros((_nInClust, 5))

        # index matching
        _indices = find_index_1D(ak_cluster_cell_ID[j].to_numpy(), idx_dict)

        _cluster[:,0] = event_dict["cluster_cell_E"][event][j].to_numpy()
        _cluster[:,1] = geo_dict["cell_geo_eta"][_indices]
        _cluster[:,2] = geo_dict["cell_geo_phi"][_indices]
        _cluster[:,3] = geo_dict["cell_geo_rPerp"][_indices]
        _cluster[:,4] = geo_dict["cell_geo_sampling"][_indices]

        clusters.append(_cluster)

    print('Number of Clusters: '+str(len(clusters)))
    print('Number of cells in each Cluster:') 
    for n in range(len(clusters)):
        print(len(clusters[n])) #okay cool good to know
    
    barrel_layer_rPerp = np.array([1540., 1733., 1930., 2450., 3010., 3630.]) #Cells in the same sampling layer (eg EMB1) have different rPerp values. these numbers were from a previous data exploration notebook where I found a statistical average. because there is no rPerp for tracks I had to just insert an average value for the different barrel layers - hence why it only works currently for central tracks. will be updated!! (rperp is the distance from the beam line)

    n_tracks = event_dict['nTrack'][event]
    print('Number of tracks: '+str(n_tracks))

    tracks = []
    for t in range(n_tracks):    #in a loop now incase there are multiple tracks!
        track = np.zeros(12)
        i = 0
        for _key in track_branches:
            track[i] = track_dict[_key][event][t]
            i += 1
        track = np.reshape(track, (6,2))
        trackP = np.full(6, event_dict['trackP'][event][t])
        track = np.column_stack((trackP, track, barrel_layer_rPerp, np.array([1,2,3,12,13,14])))#these numbers are integers that identify the calorimeters!
        tracks.append(track)
        
    #convert data:
    cluster_xyz = []
    for i in range(len(clusters)):
        _xyz = to_xyz(clusters[i][:,1:4])
        cluster_xyz.append(_xyz)

    tracks_xyz = []
    for i in range(len(tracks)):
        _xyz = to_xyz(tracks[i][:,1:4])
        tracks_xyz.append(_xyz)
        
    plottracks(cluster_xyz, tracks_xyz)