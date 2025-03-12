#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:49:57 2025

@author: fenaux
"""

import numpy as np
import supervision as sv

from matplotlib import pyplot as plt

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.Color.BLUE,
    thickness=2
)
"""ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.DEFAULT,
    color_lookup='TRACK',
    thickness=2
)"""
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    thickness=2
)

def plot_tracks(source_video_path: str, dict_file: str,
              target_video_path: str,
              start: int = 0, end: int = -1):# -> Iterator[np.ndarray]:
    """
    après la détection des joueurs, modification pour ne rechercher les équipes que si
    une nouvelle trace est apparue

    """
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    #frames = np.unique(inframe)
    bboxes = bboxes_[:,1:5]
    in_pitch = data_dict['in_pitch'].astype(np.bool_)
    track_id = data_dict['track_ids'].astype(np.int16)
    
    inframe = inframe[in_pitch]
    bboxes = bboxes[in_pitch]
    track_id = track_id[in_pitch]
    
    in_track = (track_id > -1)
    inframe = inframe[in_track]
    bboxes = bboxes[in_track]
    track_id = track_id[in_track]
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
        
            in_i_frame = np.where(inframe== i_frame)[0]

            boxes_in_frame = bboxes[in_i_frame]
            
            detections = sv.Detections(xyxy = boxes_in_frame,
                                          class_id=track_id[in_i_frame].astype(np.int16))#,
                                          #tracker_id = track_id[in_i_frame])
        
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections)

            sink.write_frame(annotated_frame)