from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

import mchmm as mc

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        #self.reducer = umap.UMAP(n_components=3)
        #self.cluster_model = KMeans(n_clusters=3)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops_loc: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops_loc)
        #projections = self.reducer.fit_transform(data)
        
        best_balence = 1e6
        for i_reduc in tqdm(range(10)):
            reduc = umap.UMAP(n_components=3)
            projections = reduc.fit_transform(data)
            for i in range(50):
                tunning = KMeans(n_clusters=3, random_state=i)
                labs  = tunning.fit_predict(projections)
                _, cnts = np.unique(labs, return_counts=True)
                sorted_cnts = np.sort(cnts)
                if (sorted_cnts[2] - sorted_cnts[1]) < best_balence:
                    self.reducer = reduc
                    self.cluster_model = tunning
                    best_balence = sorted_cnts[2] - sorted_cnts[1]
                    
        print(f"best_balence {int(best_balence)}")
        
        projections = self.reducer.transform(data)
        _, cnts = np.unique(self.cluster_model.predict(projections), return_counts=True)
        self.cluster_few = np.argmin(cnts) # so that label 2 is for referees
        # reodered labels are obtained with self.cluster_order[labs]

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        cluster_labels = self.cluster_model.predict(projections)
        if self.cluster_few != 2:
            old_2 = np.where(cluster_labels == 2)
            old_few = np.where(cluster_labels == self.cluster_few)
            cluster_labels[old_few] = 2 # so that label 2 is for referees
            cluster_labels[old_2] = self.cluster_few 
        return cluster_labels
    
def HMMarkov(unique_track_ids, track_ids_, team_id_):
    
    for track_id in unique_track_ids:
        
        in_track = np.where(track_ids_ == track_id)[0]
        old_team = team_id_[in_track]
        id_in_track, cnt_in_track = np.unique(team_id_[in_track], return_counts=True)
        cnt0 = cnt_in_track[id_in_track==0] if (id_in_track==0).any() else 0
        cnt1 = cnt_in_track[id_in_track==1] if (id_in_track==1).any() else 0
        cnt2 = cnt_in_track[id_in_track==2] if (id_in_track==2).any() else 0
        
        do_hmm = False
        #if (cnt0 > 0).size > 0 and (cnt1 > 0).size > 0: # old criteria 
        if max(cnt0, cnt1) > in_track.size / 5:
            # une meilleure décision pourrait être prise avec un dbscan
            # avec une longeur mini pour le cluster exemple track 36 de lourdes tarbes
            #if min(cnt0, cnt1) > max(cnt0, cnt1) / 6: #3: " old criteria
            do_hmm = True
            add_first= False
            work = old_team.copy()
            if np.unique(old_team).size < 3:
                add_first=True
                index_to_add = np.setdiff1d(np.arange(3), np.unique(old_team))
                work = np.insert(work, 0, index_to_add)
            """
            if min(cnt0, cnt1) > in_track.size / 7:
                
                #X = np.column_stack((np.arange(len(in_track)), team_id_[in_track])).astype(np.float32)
                #clust = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(X)
                # single et ward marchaient moins bien
                # il n'y a pas de garantie que le team_id est unique dans un cluster
                # il faudra peut-être itéré
                # in each cluster we apply "winner takes all"
                # as labels 2 (refereees) are hasardous only "team's labels are considered
        
                atp = np.eye(3)        
                atp[0,1] = atp[1,0] = min(0.01, 2 / len(in_track)) # no more than 2 switch
                atp[:2,2] = 1e-9 # when players in track referee ident is always a mistake
                atp[0,0] = atp[1,1]= 1 - atp[0,1:].sum()
                atp[2] = [0.45, 0.45, 0.1]
                
                aep = np.eye(3)
                aep[0,0] = aep[1,1] = 0.8
                aep[0,1] = aep[1,0] = aep[:2,2] = 0.1 # guess for possible mistakes
                
            elif cnt2 > in_track.size * 0.6:
                atp = np.eye(3)        
                atp[0,1] = atp[1,0] = min(0.01, 2 / len(in_track)) # no more than 2 switch
                atp[:2,2] = atp[0,1] # when players in track referee ident is always a mistake
                atp[0,0] = atp[1,1] = 1 - atp[0,1:].sum()
                atp[2] = atp[0,[1,2,0]]
                
                aep = np.eye(3)
                aep[0,0] = aep[1,1] = 0.85
                aep[0,1] = aep[1,0] = 0.05 # guess for possible mistakes
                aep[:2,2] = 0.1
                aep[2] = [0.05, 0.05, 0.9]"""
                
            if cnt2 > in_track.size * 0.5:
                atp = np.eye(3)        
                atp[0,1] = atp[1,0] = min(0.01, 2 / len(in_track)) # no more than 2 switch
                atp[:2,2] = atp[0,1] # when players in track referee ident is always a mistake
                atp[0,0] = atp[1,1] = 1 - atp[0,1:].sum()
                atp[2] = atp[0,[1,2,0]]
                
                aep = np.eye(3)
                aep[0,0] = aep[1,1] = 0.9#0.85
                aep[0,1] = aep[1,0] = 0.05 # guess for possible mistakes
                aep[:2,2] = 0.05#0.1
                aep[2] = [0.1, 0.1, 0.8]
                
            elif min(cnt0, cnt1) > in_track.size / 7:
                
                #X = np.column_stack((np.arange(len(in_track)), team_id_[in_track])).astype(np.float32)
                #clust = AgglomerativeClustering(n_clusters=2, linkage='complete').fit(X)
                # single et ward marchaient moins bien
                # il n'y a pas de garantie que le team_id est unique dans un cluster
                # il faudra peut-être itéré
                # in each cluster we apply "winner takes all"
                # as labels 2 (refereees) are hasardous only "team's labels are considered
        
                atp = np.eye(3)        
                atp[0,1] = atp[1,0] = min(0.01, 2 / len(in_track)) # no more than 2 switch
                atp[:2,2] = 1e-9 # when players in track referee ident is always a mistake
                atp[0,0] = atp[1,1]= 1 - atp[0,1:].sum()
                atp[2] = [0.45, 0.45, 0.1]
                
                aep = np.eye(3)
                aep[0,0] = aep[1,1] = 0.8
                aep[0,1] = aep[1,0] = aep[:2,2] = 0.1 # guess for possible mistakes
                
            else: do_hmm = False
            
        if do_hmm:
            a = mc.HiddenMarkovModel().from_seq(work, work)
            vs, vsi = a.viterbi(obs_seq=work, tp=atp, ep=aep)
            
            if add_first: vs = np.delete(vs, 0)
            
            
            i_new_track = track_ids_.max() + 1
            i_change = np.where( np.diff(vs) != 0)[0]
            if len(i_change) >= 2: 
                print('too much change')
                while True:
                    candidate = []
                    for i in range(3):
                        flag_i = np.where(vs==i,1,0)
                        in_i = np.where(np.diff(flag_i) > 0)[0]
                        out_i = np.where(np.diff(flag_i) < 0)[0]
                        #if vs[0]== i : in_i = np.insert(in_i, 0,-1)
                        #else: out_i = np.insert(out_i, 0,-1)
                        if vs[0] != i: out_i = np.insert(out_i, 0,-1)
                        if vs[-1]==i : out_i = np.append(out_i, flag_i.size)
                        else: in_i = np.append(in_i, flag_i.size)
                        for pif, paf in zip(out_i, in_i):
                            candidate.append([i, pif, paf])
                    candidate = np.array(candidate)
                    if candidate.size ==0: break
                    dt_candidate = np.diff(candidate[:,1:], axis=1).squeeze()
                    if dt_candidate.min() > 40: break
                    i_candidate = np.argmin(dt_candidate)
                    
                    start, end = candidate[i_candidate,1:]
                    start = start + 1
                    end = min(end +1, flag_i.size)
                    
                    vs[start:end] = candidate[i_candidate,0]
                    if np.unique(vs).size == 1: break # only one team_id
            
            team_id_[in_track] = vs.copy()
                
            for increment, ich in enumerate(i_change):
                track_ids_[in_track[ich+1:]] = i_new_track + increment
            
        if not do_hmm:
            team_id_[in_track] = id_in_track[np.argmax(cnt_in_track)]
            
    return track_ids_.copy(), team_id_.copy()
