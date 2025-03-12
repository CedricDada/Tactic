#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:17:55 2025

@author: fenaux
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import supervision as sv
import numpy as np
from matplotlib import pyplot as plt

def draw_pitch(
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    scale: float = 40
) -> np.ndarray:       

    vertices = [(0,0), (0,15), (14,0), (14,15), (28,0), (28,15),
                (0, 7.5 - 2.45), (0, 7.5 + 2.45),
                (5.8, 7.5 - 2.45), (5.8, 7.5 + 2.45),
                (28, 7.5 - 2.45), (28, 7.5 + 2.45),
                (28 - 5.8, 7.5 - 2.45), (28 - 5.8, 7.5 + 2.45),
                (14, 7.5 - 1.8), (14, 7.5 + 1.8)]
    
    centers = [(5.8, 7.5), (28 - 5.8, 7.5), (14, 7.5)]
                     
    edges =[(0, 1), (0, 4), (1, 5), (4, 5),
        (6, 8), (7, 9), (8, 9),
        (10, 12), (11, 13), (12, 13),
        (2,14), (3, 15)]
    
    angles = [180, 0]
    
    
    width = 15
    length = 28
    radius = 1.8
    
    scaled_width = int(width * scale)
    scaled_length = int(length * scale)
    scaled_radius = int(radius * scale)
    
    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)
    
    for start, end in edges:
        point1 = (int(vertices[start][0] * scale) + padding,
                  int(vertices[start][1] * scale) + padding)
        point2 = (int(vertices[end][0] * scale) + padding,
                  int(vertices[end][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
            
    for center, angle in zip(centers[:2], angles):
        pt = (int(center[0] * scale) + padding,
                  int(center[1] * scale) + padding)
        axes = (scaled_radius, scaled_radius)
        cv2.ellipse(img=pitch_image,
                    center=pt, axes=axes,
                    angle=angle, startAngle=90, endAngle=270,
                    color=line_color.as_bgr(), thickness=line_thickness)
        
    center = centers[2]
    pt = (int(center[0] * scale) + padding,
              int(center[1] * scale) + padding)
    
    cv2.circle(
        img=pitch_image,
        center=pt,
        radius=scaled_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )
        
    return pitch_image

def draw_points_on_pitch(
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 40,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config (SoccerPitchConfiguration): Configuration object containing the
            dimensions and layout of the pitch.
        xy (np.ndarray): Array of points to be drawn, with each point represented by
            its (x, y) coordinates.
        face_color (sv.Color, optional): Color of the point faces.
            Defaults to sv.Color.RED.
        edge_color (sv.Color, optional): Color of the point edges.
            Defaults to sv.Color.BLACK.
        radius (int, optional): Radius of the points in pixels.
            Defaults to 10.
        thickness (int, optional): Thickness of the point edges in pixels.
            Defaults to 2.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        scale (float, optional): Scaling factor for the pitch dimensions.
            Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image to draw points on.
            If None, a new pitch will be created. Defaults to None.

    Returns:
        np.ndarray: Image of the soccer pitch with points drawn on it.
    """
    if pitch is None:
        pitch = draw_pitch(
            padding=padding,
            scale=scale
        )

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return np.flipud(pitch)

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.Color.BLUE,
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.Color.YELLOW,
    thickness=2
)

def run_radar(source_video_path: str, dict_file: str,
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
    players = data_dict['xy'].astype(np.int16)
    
    inframe = inframe[in_pitch]
    bboxes = bboxes[in_pitch]
    track_id = track_id[in_pitch]
    players = players[in_pitch]
    
    in_track = (track_id > -1)
    inframe = inframe[in_track]
    bboxes = bboxes[in_track]
    track_id = track_id[in_track]
    players = players[in_track]
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
   
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            in_i_frame = np.where(inframe== i_frame)[0]
            players_in_frame = players[in_i_frame]
            boxes_in_frame = bboxes[in_i_frame]
            
            """
            detections_in = sv.Detections(xyxy = boxes_in_frame[are_in_pitch_frame],
                                          class_id=np.zeros(are_in_pitch_frame.sum()))
            detections_out = sv.Detections(xyxy = boxes_in_frame[are_out_pitch_frame],
                                           class_id=np.zeros(are_out_pitch_frame.sum()))
            
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections_in)
            annotated_frame = BOX_ANNOTATOR.annotate(
                annotated_frame, detections_out)"""
    
            radar = draw_points_on_pitch(players_in_frame)
            
            h, w, _ = frame.shape
            radar = sv.resize_image(radar, (w // 2, h // 2))
            radar_h, radar_w, _ = radar.shape
            rect = sv.Rect(
                x=w // 2 - radar_w // 2,
                y=h - radar_h,
                width=radar_w,
                height=radar_h
            )
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
            sink.write_frame(annotated_frame)