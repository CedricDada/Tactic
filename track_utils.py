#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:27:47 2025

@author: fenaux
"""

import numpy as np
import supervision as sv

from scipy.spatial import distance

def run_sv_tracker(boxes_file: str) :
    """
    Run tracking on boxes with bytetrack

    Args:
         (str): Path to detections 
        

    """
    
    bboxes_ = np.load(boxes_file)
    inframe = bboxes_[:,0].astype(np.int16)
    frames = np.unique(inframe)
    confs = bboxes_[:,5]
    bboxes = bboxes_[:,1:5]
    
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    tracker.reset() # needed to have tracks from 1 if video was processed before
    keep_track = np.array([])
    for i_frame in frames:
        in_i_frame = np.where(inframe== i_frame)[0]
        n_in_i_frame = len(in_i_frame)
        boxes_in_i_frame = bboxes[in_i_frame]
        conf_in_i_frame = confs[in_i_frame]
        
        detections = sv.Detections(xyxy = boxes_in_i_frame,
                                      class_id=np.zeros(n_in_i_frame),
                                      confidence=conf_in_i_frame,
                                      data={'0':np.arange(n_in_i_frame)})
        
        detections = tracker.update_with_detections(detections)

        tracks = - np.ones(n_in_i_frame)
        tracks[detections.data['0']] = detections.tracker_id.copy()
        keep_track = np.append(keep_track, tracks)
        #labels = [str(tracker_id) for tracker_id in detections.tracker_id]
    
    return {'bboxes':bboxes_,'track_ids':keep_track}
 

def box_and_track(boxes_file:str, track_file:str, dict_file:str, ConfOnly:bool = True):
    """
    
    Parameters
    ----------
    boxes_file:str name of file where detections are stored numpy array each row
        [i_frame, xtop, ytop, xbottom, ybottom, confidence]
    track_file:str name of file where tracker output are stored numpy array each row
            [i_frame, xtop, ytop, xbottom, ybottom, confidence, track_id]
    boxes may be different in boxes_file an track_file due to kalmann filtering
    ConfOnly : bool, optional
        DESCRIPTION. The default is True. If True pairing of boxes only relies on confidence
        should be set to true for BotSort False for DeepOcSort
        
    dict_file : file where to save results
    ['bboxes'] like boxes_file with kalmann improvement
    ['track_ids'] ids from track_file, id is -1 if not in a track
    Returns
    -------
    None.

    """

    tracks = np.load(track_file)
    if len(tracks.shape) == 1:
        tracks = tracks.reshape(-1,8)
    if np.array_equal(tracks[:,5], tracks[:,6]):
        tracks = np.delete(tracks, 6, axis=1)
        
    tracks_ids = tracks[:,-1]
    
    bboxes_ = np.load(boxes_file)
    inframe = bboxes_[:,0].astype(np.int16)
    frames = np.unique(inframe)
    
    in_which_track = -np.ones(len(bboxes_))
    for i_frame in frames:
        
        in_i_frame = np.where(inframe== i_frame)[0]
        n_in_i_frame = len(in_i_frame)
        boxes_in_i_frame = bboxes_[in_i_frame,1:]
        
        in_i_frame_track = np.where(tracks[:,0] == i_frame)[0]
        boxes_in_i_frame_track = tracks[in_i_frame_track,1:-1]
        
        if ConfOnly:
            conf_in_i_frame = boxes_in_i_frame[:,-1].reshape(-1,1)
            conf_in_i_frame_track = boxes_in_i_frame_track[:,-1].reshape(-1,1)
            pairdist = distance.cdist(conf_in_i_frame, conf_in_i_frame_track, 'euclidean')
            
        else:       
            pairdist = distance.cdist(boxes_in_i_frame, boxes_in_i_frame_track, 'euclidean')
                
        pairs = np.where(pairdist == 0)
      
        in_which_track[in_i_frame[pairs[0]]] = tracks_ids[in_i_frame_track[pairs[1]]]
        
        if i_frame % 100 == 0 and i_frame > 1:
            print( np.linalg.norm(
                bboxes_[in_i_frame[pairs[0]],1:-1] - tracks[in_i_frame_track[pairs[1]],1:-2],
                axis=1).max()
                )

        if ConfOnly: # to keep improvement due to kalmann filtering
            bboxes_[in_i_frame[pairs[0]],1:] = tracks[in_i_frame_track[pairs[1]],1:-1]

    
    deep_dict = {'bboxes':bboxes_,'track_ids':in_which_track.astype(np.int16)}
    np.save(dict_file, deep_dict)
     

def track_in_pitch(dict_file):
    
    data_dict = np.load(dict_file, allow_pickle=True).item()
    track_ids = data_dict['track_ids']
    xy = data_dict['xy']
    
    is_on_pitch = np.zeros(len(xy))
    i_tracks = np.unique(track_ids)
    i_tracks = i_tracks[i_tracks > - 1]
    
    in_pitch_x = (xy[:,0] > -2) * (xy[:,0] < 30)
    in_pitch_y = (xy[:,1] > -0.5) * (xy[:,1] < 15.5)
    in_pitch_flag = in_pitch_x * in_pitch_y
    for i_track in i_tracks:
        
        in_i_track = np.where(track_ids== i_track)[0]
        in_pitch_ratio = in_pitch_flag[in_i_track].mean()
        if in_pitch_ratio > 0.5:
            is_on_pitch[in_i_track] = 1
    
    data_dict['in_pitch'] = is_on_pitch.astype('bool_')
    np.save(dict_file, data_dict)
       
def StartsEnds(dict_file:str, pitch_only:bool = False):
    track_dict = np.load(dict_file, allow_pickle=True).item()
    
    bboxes_ = track_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy']
    track_ids = track_dict['track_ids']
    if pitch_only: 
        is_on_pitch = track_dict['in_pitch']
        track_ids = track_ids[is_on_pitch]
        inframe = inframe[is_on_pitch]
        xy = xy[is_on_pitch]
        
    i_tracks = np.unique(track_ids)
    i_tracks = i_tracks[i_tracks > -1]
    starts_ends = []
    for i_track in i_tracks:
        in_i_track = np.where(track_ids == i_track)[0]
        starts_ends.append([inframe[in_i_track].min(), inframe[in_i_track].max(), i_track])
        
    from matplotlib import pyplot as plt
    plt.scatter(xy[:,0], xy[:,1], c=track_ids, 
                norm = plt.Normalize(vmin=0, vmax=track_ids.max()), cmap="nipy_spectral" ,s=0.2)
    plt.show()
    return np.array(starts_ends)
    
    