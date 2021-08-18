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

def find_closest2(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel,  clusPt_sel, clusmass_sel, POI, part2, SAME=False): #POI stands for particle of interest
    
    print('Finding cluster closest to particle', POI)
    #loop thru
    mindist_indices = []
    selec_inds = []
    for i in range(len(trutheta_sel)): 
        
        #select the poi:
        n_mask = truthID_sel[i] == POI
        #and the other particle:
        o_mask = truthID_sel[i] == part2

        #then get deltaR for poi
        partvec = ak.zip({
            "pt": truthpt_sel[i][n_mask],
            "eta": trutheta_sel[i][n_mask],
            "phi": truthphi_sel[i][n_mask],
            "mass": truthmass_sel[i][n_mask],
        })

        part4Dvec = ak.with_name(partvec, "Momentum4D")
        
        #and the other particle:
        partvec2 = ak.zip({
            "pt": truthpt_sel[i][o_mask],
            "eta": trutheta_sel[i][o_mask],
            "phi": truthphi_sel[i][o_mask],
            "mass": truthmass_sel[i][o_mask],
        })

        part4Dvec2 = ak.with_name(partvec2, "Momentum4D") 
        
        #and for the cluster
        deltaR = []
        deltaR2 = []
        for j in range(len(clusEta_sel[i])):
            clusvec = ak.zip({
                "pt": clusPt_sel[i][j],
                "eta": clusEta_sel[i][j],
                "phi": clusPhi_sel[i][j],
                "mass": clusmass_sel[i][j]})
            clus4Dvec = ak.with_name(clusvec, "Momentum4D")

            deltaR.append(part4Dvec.deltaR(clus4Dvec))
            deltaR2.append(part4Dvec2.deltaR(clus4Dvec))
            if deltaR == []:
                print('exception: empty deltaR') 
                print(part4Dvec, clus4Dvec, part4Dvec.deltaR(clus4Dvec), deltaR, i, j)
                
            ind = np.argmin(np.array(ak.flatten(deltaR)))
            ind2 = np.argmin(np.array(ak.flatten(deltaR2)))
            
        if (ind == ind2) == SAME:
            selec_inds.append(i)
            mindist_indices.append(ind)
    
    print(len(mindist_indices))
    print('selec inds', len(selec_inds))
    #probably janky but use only the ones that were in mindist
    truthID_sel_cco = []
    trutheta_sel_cco = [] #also removes all the events/truth particles in the events with no clusters
    truthphi_sel_cco = []
    truthpt_sel_cco = []
    truthmass_sel_cco = []
    clusE_sel_cco = []
    clusEta_sel_cco =[]
    clusPhi_sel_cco = []
    clusPt_sel_cco = []
    clusmass_sel_cco = []
    mindist_inds_cco = []
    for i in selec_inds: 
        truthID_sel_cco.append(truthID_sel[i])
        trutheta_sel_cco.append(trutheta_sel[i])
        truthphi_sel_cco.append(truthphi_sel[i])
        truthpt_sel_cco.append(truthpt_sel[i])
        truthmass_sel_cco.append(truthmass_sel[i])
        clusE_sel_cco.append(clusE_sel[i])
        clusEta_sel_cco.append(clusEta_sel[i])
        clusPhi_sel_cco.append(clusPhi_sel[i])
        clusPt_sel_cco.append(clusPt_sel[i])
        clusmass_sel_cco.append(clusmass_sel[i])
        #mindist_inds_cco.append(mindist_indices[i])
    
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
    for i in range(len(clusEta_sel_cco)): 
        if len(clusEta_sel_cco[i]) != 0: 
            mindist_indices_full.append(mindist_indices[i])
            truthID_sel_full.append(truthID_sel_cco[i])
            trutheta_sel_full.append(trutheta_sel_cco[i])
            truthphi_sel_full.append(truthphi_sel_cco[i])
            truthpt_sel_full.append(truthpt_sel_cco[i])
            truthmass_sel_full.append(truthmass_sel_cco[i])
            clusE_sel_full.append(clusE_sel_cco[i])
            clusEta_sel_full.append(clusEta_sel_cco[i])
            clusPhi_sel_full.append(clusPhi_sel_cco[i])
            clusPt_sel_full.append(clusPt_sel_cco[i])
            clusmass_sel_full.append(clusmass_sel_cco[i])
            full_inds.append(selec_inds[i])
      
   # print(len(mindist_indices_full), mindist_indices_full)
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

def find_closest_rho(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel, clusPt_sel, clusmass_sel, SAME = False): #I knowww I'm lazy I should turn these all into one function... oh well
    
    print('Finding cluster closest to particle', 111)
    #loop thru
    mindist_indices = []
    selec_inds = []
    for i in range(len(trutheta_sel)): 
        
        #select the poi:
        n_mask = truthID_sel[i] == 111
        #and the other particle:
        o_mask = np.logical_or(truthID_sel[i] == 211, truthID_sel[i] == -211)

        print(o_mask)
        #then get deltaR for poi
        partvec = ak.zip({
            "pt": truthpt_sel[i][n_mask],
            "eta": trutheta_sel[i][n_mask],
            "phi": truthphi_sel[i][n_mask],
            "mass": truthmass_sel[i][n_mask],
        })

        part4Dvec = ak.with_name(partvec, "Momentum4D")
        
        #and the other particle:
        partvec2 = ak.zip({
            "pt": truthpt_sel[i][o_mask],
            "eta": trutheta_sel[i][o_mask],
            "phi": truthphi_sel[i][o_mask],
            "mass": truthmass_sel[i][o_mask],
        })

        part4Dvec2 = ak.with_name(partvec2, "Momentum4D") 
        
        #and for the cluster
        deltaR = []
        deltaR2 = []
        for j in range(len(clusEta_sel[i])):
            clusvec = ak.zip({
                "pt": clusPt_sel[i][j],
                "eta": clusEta_sel[i][j],
                "phi": clusPhi_sel[i][j],
                "mass": clusmass_sel[i][j]})
            clus4Dvec = ak.with_name(clusvec, "Momentum4D")

            deltaR.append(part4Dvec.deltaR(clus4Dvec))
            deltaR2.append(part4Dvec2.deltaR(clus4Dvec))
            if deltaR == []:
                print('exception: empty deltaR') 
                print(part4Dvec, clus4Dvec, part4Dvec.deltaR(clus4Dvec), deltaR, i, j)
                
            ind = np.argmin(np.array(ak.flatten(deltaR)))
            ind2 = np.argmin(np.array(ak.flatten(deltaR2)))
            
        if (ind == ind2) == SAME: #only change lol; so if they are equal and you want to exclude those, then leave same as false, but if you do want the cluster to be the same just set same to true 
            selec_inds.append(i)
            mindist_indices.append(ind)
    
    print(len(mindist_indices))
    print('selec inds', len(selec_inds))
    #probably janky but use only the ones that were in mindist
    truthID_sel_cco = []
    trutheta_sel_cco = [] #also removes all the events/truth particles in the events with no clusters
    truthphi_sel_cco = []
    truthpt_sel_cco = []
    truthmass_sel_cco = []
    clusE_sel_cco = []
    clusEta_sel_cco =[]
    clusPhi_sel_cco = []
    clusPt_sel_cco = []
    clusmass_sel_cco = []
    mindist_inds_cco = []
    for i in selec_inds: 
        truthID_sel_cco.append(truthID_sel[i])
        trutheta_sel_cco.append(trutheta_sel[i])
        truthphi_sel_cco.append(truthphi_sel[i])
        truthpt_sel_cco.append(truthpt_sel[i])
        truthmass_sel_cco.append(truthmass_sel[i])
        clusE_sel_cco.append(clusE_sel[i])
        clusEta_sel_cco.append(clusEta_sel[i])
        clusPhi_sel_cco.append(clusPhi_sel[i])
        clusPt_sel_cco.append(clusPt_sel[i])
        clusmass_sel_cco.append(clusmass_sel[i])
        #mindist_inds_cco.append(mindist_indices[i])
    
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
    for i in range(len(clusEta_sel_cco)): 
        if len(clusEta_sel_cco[i]) != 0: 
            mindist_indices_full.append(mindist_indices[i])
            truthID_sel_full.append(truthID_sel_cco[i])
            trutheta_sel_full.append(trutheta_sel_cco[i])
            truthphi_sel_full.append(truthphi_sel_cco[i])
            truthpt_sel_full.append(truthpt_sel_cco[i])
            truthmass_sel_full.append(truthmass_sel_cco[i])
            clusE_sel_full.append(clusE_sel_cco[i])
            clusEta_sel_full.append(clusEta_sel_cco[i])
            clusPhi_sel_full.append(clusPhi_sel_cco[i])
            clusPt_sel_full.append(clusPt_sel_cco[i])
            clusmass_sel_full.append(clusmass_sel_cco[i])
            full_inds.append(selec_inds[i])
      
   # print(len(mindist_indices_full), mindist_indices_full)
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


def find_closest(truthID_sel, trutheta_sel, truthphi_sel, truthpt_sel, truthmass_sel, clusE_sel, clusEta_sel, clusPhi_sel,  clusPt_sel, clusmass_sel, POI): #POI stands for particle of interest
    
    print('Finding cluster closest to particle', POI)
    #loop thru
    mindist_indices = []
    for i in range(len(trutheta_sel)): 
        
        #select the poi:
        n_mask = truthID_sel[i] == POI

        #then get deltaR for poi
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