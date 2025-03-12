#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:27:18 2025

@author: fenaux
"""

import numpy as np
from matplotlib import pyplot as plt

import os
import time
import gc

import cv2
import supervision as sv

from typing import List

from func_players_batch import func_box
from func_in_pitch import on_pitch
from pitch_utils import draw_points_on_pitch, run_radar
#from track_utils import run_sv_tracker
from track_utils import track_in_pitch, box_and_track, StartsEnds
from render_track import plot_tracks
from team import TeamClassifier, HMMarkov

device = 'cuda'

video_in = "../ffb/CFBB vs UNION TARBES LOURDES PYRENEES BASKET Men's Pro Basketball - Tactical.mp4"
homog_file = '../pitch/Hs_supt1.npy'
pitch_file = '../pitch/pitch.npy'
video_track = 'video_clip.mp4'


boxes_file = 'boxes.npy'
track_file = 'tracks_clip.npy'
dict_file = 'clip_dict_3.npy'
pitch = np.load(pitch_file)
corners = pitch[[0,1,4,5]].copy()
lines = [[0,1], [0,2],[1,2],[2,3]]

if not os.path.exists(boxes_file):
    func_box(video_in, boxes_file, start_frame=100_001, end_frame=100_001+2000)

#byte_dict = run_sv_tracker(boxes_file)

#box_and_track(boxes_file, track_file, dict_file)
#on_pitch(dict_file, homog_file)
#track_in_pitch(dict_file)
#starts_ends = StartsEnds(dict_file, pitch_only=True)

from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
def ChainTrack(dict_file:str, starts_ends):
    
    track_dict = np.load(dict_file, allow_pickle=True).item()
    
    bboxes_ = track_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    xy = track_dict['xy']
    track_ids = track_dict['track_ids']
    
    xys = []
    for startT, endT, idT in starts_ends:
        in_track = np.where(track_ids==idT)[0]
        xy_in_track = xy[in_track]
        xys.append(np.append(xy_in_track[0], xy_in_track[-1]))
    xys = np.array(xys) / 15
    debs = np.column_stack((starts_ends[:,0] / 30, xys[:,:2]))
    fins = np.column_stack((starts_ends[:,1] / 30, xys[:,2:]))
    
    
    
    first_end = starts_ends[:,1].min()
    last_start = starts_ends[:,0].max()
    
    to_keep = np.where( (starts_ends[:,0] > first_end) * 
                        (starts_ends[:,1] < last_start) )[0]
    debs = debs[to_keep]
    fins = fins[to_keep]
    track_ids = starts_ends[to_keep]
    
    StartEndDiff= debs[:,0].reshape(-1,1) - fins[:,0].reshape(1,-1)
    StartBeforeEnd = StartEndDiff <= 0
    StartTooLate = StartEndDiff > 2
    BadStart = StartBeforeEnd + StartTooLate
    pairdist = distance.cdist(debs, fins, 'euclidean')
    
    pairdist[BadStart] = 1e6
    pairs = np.argmin(pairdist, axis=1)
    
    pairsMinDist = pairdist[(np.arange(len(pairs)),pairs)]
    invalids = np.where(pairsMinDist==1e6)[0]
    idx = np.delete( np.arange(len(pairs)), invalids)
    pairs = np.delete( pairs,invalids)
    pairsMinDist = pairdist[(idx,pairs)]
    
    return xys

#xys = ChainTrack(dict_file, starts_ends.copy())
        
        
#run_player_tracking(boxes_file)

#run_radar(video_track, dict_file,
#              'radar_bot.mp4')#,
              #start=100_001, end=100_001 + 2000)
              
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def crop_track(track_id, video_in, tracks, bboxes, inframe, stride, init_frame):
    bboxes = np.clip(bboxes,0,None)
    in_track = np.where(tracks == track_id)[0]
    in_track_frames = inframe[in_track]
    in_track_boxes = bboxes[in_track]
        
    cap = cv2.VideoCapture()
    cap.open(video_in )
    
    crops = []
    for i_frame, box in zip(in_track_frames[::stride], in_track_boxes[::stride]):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i_frame + init_frame)
        ret, frame = cap.read()
        #if i + first_in_track in in_track_frames:
        
        detections = sv.Detections(xyxy = box.reshape(1,4))
        crops += get_crops(frame, detections)
        
    return crops
        

track_dict = np.load(dict_file, allow_pickle=True).item()

bboxes_ = track_dict['bboxes']
inframe_ = bboxes_[:,0].astype(np.int16)
boxes_ = bboxes_[:,1:5]
xy_ = track_dict['xy']
track_ids_ = track_dict['track_ids']

not_in_pitch = np.logical_and( np.logical_not(track_dict['in_pitch']),
                              track_ids_ > 0)
track_ids_[not_in_pitch] *= -1
track_ids_[not_in_pitch] -= 1

idx_tracks_valid = np.where(track_ids_ > -1)[0]
track_ids = track_ids_[idx_tracks_valid]

vits = []
unique_track_ids = np.unique(track_ids)
for i_track in unique_track_ids:
    in_track = np.where(track_ids_ == i_track)[0]
    dxdy = np.gradient(xy_[in_track], inframe_[in_track], axis=0)
    ds = np.linalg.norm(dxdy, axis=1)
    vits.append(np.quantile(ds, 0.9)) # / inframe_[in_track].ptp())
vits = np.array(vits)

slow_threshold = 0.14 # with 30 fps
where_slow = np.where(vits < slow_threshold)[0]
# vits_slow = vits[vits < slow_threshold]
# sort_idx_slow = idx_tracks[where_slow[np.argsort(vits_slow)]]
unique_track_ids = np.delete(unique_track_ids, where_slow)
vits = np.delete(vits, where_slow)
    

# to initialze classifier we take a strict definition of pitch
# in order to exclude coaches or public
is_in_pitch_x = (xy_[:,0] > 0) * (xy_[:,0] < 28)
is_in_pitch_y = (xy_[:,1] > 0) * (xy_[:,1] < 15)
is_in_pitch = is_in_pitch_x * is_in_pitch_y

inframe = inframe_[is_in_pitch]
boxes = boxes_[is_in_pitch]
"""
stride = 50#100
start = 100_001
source_video_path = video_in
source = sv.get_video_frames_generator(
    source_path=source_video_path, start=100_001, end=100_001 + 2000, stride=stride)

crops = []
for i_frame, frame in enumerate(source):
    
    in_this_frame = np.where(inframe == i_frame * stride)[0]
    detections = sv.Detections(xyxy = boxes[in_this_frame])
    crops += get_crops(frame, detections)

t0 = time.time()
team_classifier = TeamClassifier(device=device)
team_classifier.fit(crops)


move_idx = np.hstack([np.where(track_ids_ == track_id)[0] for track_id in unique_track_ids])
inframe_move = inframe_[move_idx]
boxes_move = boxes_[move_idx]
track_ids_move = track_ids_[move_idx]
source = sv.get_video_frames_generator(
    source_path=source_video_path, start=100_001, end=100_001 + 2000)

crops = []
idx_crops = np.array([]).astype(np.int16)
boxes_ = np.clip(boxes_, 0, None)
player_team_id = np.array([])

for i_frame, frame in enumerate(source):
    in_this_frame = np.where(inframe_ == i_frame)[0]
    in_this_frame_move = in_this_frame[np.isin(track_ids_[in_this_frame], unique_track_ids)]
    detections = sv.Detections(xyxy = boxes_[in_this_frame_move])
    crops += get_crops(frame, detections)
    idx_crops = np.append(idx_crops, in_this_frame_move)
    
    if (i_frame >= 2000) or (len(crops) > 256):
        new_team_id = team_classifier.predict(crops)
        player_team_id = np.append(player_team_id, new_team_id)
        print(i_frame, len(crops))
        crops = []

if len(crops) > 0:
    new_team_id = team_classifier.predict(crops)
    player_team_id = np.append(player_team_id, new_team_id)

team_id = -np.ones(len(boxes_))
team_id[idx_crops] = player_team_id
print(time.time() - t0)
track_dict['team_id'] = team_id.astype(np.int16)
np.save(dict_file,track_dict)


track_ids_hmm, team_id_hmm = HMMarkov(unique_track_ids, 
                                      track_ids_.copy(), track_dict['team_id'].copy())

track_dict['track_ids_hmm']= track_ids_hmm.copy()
track_dict['team_id_hmm'] = team_id_hmm.copy()
np.save(dict_file,track_dict)
"""
team_id_ = track_dict['team_id']
team_id_hmm = track_dict['team_id_hmm']
track_ids_hmm = track_dict['track_ids_hmm']
show = True
if show:
    stride = 4
    for track_id in unique_track_ids:
        in_track = np.where(track_ids_ == track_id)[0]
        id_in_track, cnt_in_track = np.unique(team_id_[in_track], return_counts=True)
        cnt0 = cnt_in_track[id_in_track==0] if (id_in_track==0).any() else 0
        cnt1 = cnt_in_track[id_in_track==1] if (id_in_track==1).any() else 0
        cnt2 = cnt_in_track[id_in_track==2] if (id_in_track==2).any() else 0
        
        do_hmm_1 = (max(cnt0, cnt1) > in_track.size / 5)
        do_hmm_2 = (cnt2 > in_track.size * 0.5) or (min(cnt0, cnt1) > in_track.size / 7)
        do_hmm = do_hmm_1 and do_hmm_2
        
        print('hmm', do_hmm)
        crops = crop_track(track_id, video_in, 
                           track_ids_, boxes_, inframe_,
                           stride=stride, init_frame=100_001)
        old_team = team_id_[in_track][::stride]
        new_team = team_id_hmm[in_track][::stride]
        show_frames = inframe_[in_track][::stride]
        for crop, old, new, i_frame in zip(crops, old_team, new_team, show_frames):
            
            plt.imshow(crop[...,::-1])
            plt.axis('off')
            if do_hmm: plt.title(f"{track_id}, {new}, {i_frame}, {old}")
            else: plt.title(f"{track_id}, {new}, {i_frame}")
            plt.show()
        print(track_id, np.unique(team_id_hmm[in_track], return_counts=True))

        print( 'continue y / n')
        stopIt = True  if str(input()) == 'n' else False
        if stopIt: break

