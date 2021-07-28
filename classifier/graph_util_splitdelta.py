import uproot as ur
import awkward as ak
import numpy as np


def loadVectorBranchFlat(branchName, tree, mask):
    X = tree[branchName].array()
    return np.copy(ak.flatten(X[mask]).to_numpy())

def make_deltaR(flatID, flateta, flatphi, flatpt, flatm, clusE, partID1, partID2, singlecluster = False):
    #so have to pass it the IDs of the particles to find the distance between since it'll be different for each type of delta!!!
    cut1 = flatID == partID1
    cut2 = flatID == partID2
    
    eta_slice1 = flateta[cut1]
    eta_slice2 = flateta[cut2]

    phi_slice1 = flatphi[cut1]
    phi_slice2 = flatphi[cut2]

    pt_slice1 = flatpt[cut1]
    pt_slice2 = flatpt[cut2]

    m_slice1 = flatm[cut1]
    m_slice2 = flatm[cut2]
    
    comb1 = ak.zip({
        "pt": pt_slice1,
        "eta": eta_slice1,
        "phi": phi_slice1,
        "mass": m_slice1,
    })

    comb2 = ak.zip({
        "pt": pt_slice2,
        "eta": eta_slice2,
        "phi": phi_slice2,
        "mass": m_slice2,
    })
    
    vec4D1 = ak.with_name(comb1, "Momentum4D")
    vec4D2 = ak.with_name(comb2, "Momentum4D")
    
    deltaR = vec4D1.deltaR(vec4D2)
    
    #this part doesn't help for single cluster
    if singlecluster == False:
        longflag = []
        for i in range(len(clusE)):
            longflag.append(np.repeat(deltaR[i], len(clusE[i])))
        
        longDeltaR = ak.flatten(ak.Array(longflag))
    else:
        longDeltaR = deltaR

    return longDeltaR #should be the same length!! **this is used to get it to be the same length as clus_e so that the selection will work on it.....


def find_closest(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel,  clusPt_sel, clusmass_sel, POI): #POI stands for particle of interest
    
    print('Finding cluster closest to particle', POI)
    #loop thru
    mindist_indices = []
    for i in range(len(trutheta_sel)): 
        
        #select the neutron:
        n_mask = truthID_sel[i] == POI

        #then get deltaR for n
        partvec = ak.zip({
            "pt": truthpt_sel[i][n_mask],
            "eta": trutheta_sel[i][n_mask],
            "phi": truthphi_sel[i][n_mask],
            "mass": truthmass_sel[i][n_mask],
        })

        part4Dvec = ak.with_name(partvec, "Momentum4D")

        #and for the cluster
        deltaR = []
        for j in range(len(clusEta_sel[i])):
            clusvec = ak.zip({
                "pt": clusPt_sel[i][j],
                "eta": clusEta_sel[i][j],
                "phi": clusPhi_sel[i][j],
                "mass": clusmass_sel[i][j]})
            clus4Dvec = ak.with_name(clusvec, "Momentum4D")

            deltaR.append(part4Dvec.deltaR(clus4Dvec))
            
            if deltaR == []:
                print('exception: empty deltaR') 
                print(part4Dvec, clus4Dvec, part4Dvec.deltaR(clus4Dvec), deltaR, i, j)
                
            ind = np.argmin(np.array(ak.flatten(deltaR)))
            
        mindist_indices.append(ind)
        
    #remove the empty ones
    mindist_indices_full = []
    truthID_sel_full = []
    trutheta_sel_full = [] #also removes all the events/truth particles in the events with no clusters
    truthphi_sel_full = []
    truthpt_sel_full = []
    truthmass_sel_full = []
    clusE_sel_full = []
    clusEta_sel_full =[]
    clusPhi_sel_full = []
    clusPt_sel_full = []
    clusmass_sel_full = []
    
    full_inds = []
    for i in range(len(clusEta_sel)): 
        if len(clusEta_sel[i]) != 0: 
            mindist_indices_full.append(mindist_indices[i])
            truthID_sel_full.append(truthID_sel[i])
            trutheta_sel_full.append(trutheta_sel[i])
            truthphi_sel_full.append(truthphi_sel[i])
            truthpt_sel_full.append(truthpt_sel[i])
            truthmass_sel_full.append(truthmass_sel[i])
            clusE_sel_full.append(clusE_sel[i])
            clusEta_sel_full.append(clusEta_sel[i])
            clusPhi_sel_full.append(clusPhi_sel[i])
            clusPt_sel_full.append(clusPt_sel[i])
            clusmass_sel_full.append(clusmass_sel[i])
            full_inds.append(i)
            
    clusEta_closest = []
    clusE_closest = [] #select only the closest clusters
    clusPhi_closest = []
    clusPt_closest = []
    clusmass_closest = []
    for i in range(len(clusEta_sel_full)):
        clusEta_closest.append(clusEta_sel_full[i][mindist_indices_full[i]])
        clusE_closest.append(clusE_sel_full[i][mindist_indices_full[i]])
        clusPhi_closest.append(clusPhi_sel_full[i][mindist_indices_full[i]])
        clusPt_closest.append(clusPt_sel_full[i][mindist_indices_full[i]])
        clusmass_closest.append(clusmass_sel_full[i][mindist_indices_full[i]])
       
    print('Closest clusters found.')
    return (truthID_sel_full, trutheta_sel_full, truthphi_sel_full, truthpt_sel_full, truthmass_sel_full, ak.Array(clusE_closest), ak.Array(clusEta_closest), ak.Array(clusPhi_closest), ak.Array(clusPt_closest), ak.Array(clusmass_closest), full_inds, mindist_indices_full)

    
#given a branchname, a tree from uproot, and a padLength...
#return a flattened numpy array that flattens away the event index and pads cels to padLength
#if there's no cell, add a 0 value
def loadArrayBranchFlat(branchName, tree, padLength, mask, empty=None, mindist_inds = None):
    branchInfo = tree[branchName].array()
    branchInfo = branchInfo[mask]
    
    #empty will be the list of indices that AREN'T EMPTY
    if empty is not None:
        branchInfo = [branchInfo[i] for i in empty] 
    
    if mindist_inds is not None:
        brnchinfo = []
        for i in range(len(branchInfo)):
            brnchinfo.append(branchInfo[i][mindist_inds[i]]) #then this should be selecting the closest cluster only!!!!
        #don't need to flatten in this case
        branchFlat = brnchinfo 
    else:
        branchFlat = ak.flatten(branchInfo)# we flatten the event index, to generate a list of clusters

    # pad the cell axis to the specified length
    branchFlatPad = ak.pad_none(branchFlat, padLength, axis=1)

    # # Do a deep copy to numpy so that the data is owned by numpy
    branchFlatNumpy = np.copy(branchFlatPad.to_numpy())
    
    return branchFlatNumpy

# A quick implemention of Dilia's idea for converting the geoTree into a dict
def loadGraphDictionary(tree):
    # make a global dict. this will be keyed by strings for which info you want
    globalDict = {}

    #get the information
    arrays = tree.arrays()
    keys = tree.keys()
    for key in keys:
        #skip geoID-- that's our new key
        if key == 'cell_geo_ID': 
            continue
        branchDict = {}
        # loop over the entries of the GEOID array (maybe this should be hte outer array? eh.)
        # [0] is used here and below to remove a false index
        for iter, ID in enumerate(arrays['cell_geo_ID'][0]):
            #the key is the ID, the value is whatever we iter over
            branchDict[ID] = arrays[key][0][iter] 
       

        if key == 'cell_geo_sampling':
            mask = 0
        else:
            mask = None

        branchDict[0] = mask
        branchDict[4308257264] = mask # another magic safetey number? CHECKM
            
        
        globalDict[key] = branchDict

    return globalDict


# given a list of Cell IDs and a target from the geometry tree specified in geoString
# (and given the globalDict containing the ID->info mappings)
# return a conversion of the cell IDs to whatever is requested
def convertIDToGeo(cellID, geoString, globalDict):
    # MAGIC https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
    return np.vectorize(globalDict[geoString].get)(np.nan_to_num(cellID))
                #this is just using an array to select from a dictionary
                #so if I get mindist to apply to cell_id then it should carry over to here too???