from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import getBboxWidth, getCenterOfBbox

class Tracker:
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        """Returns the detections from the model for each frame"""
        # Batch size to account for memory issues.
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Track each object in each frame of the video.
        In each frame, for each object, we store the bounding box in a dict.
        Store the data in a convenient format shown below and return the frame.
        We are also going to save the tracks dictionary so that we don't have to track again and again.
        """
        
        # Loading tracks from pickle file.
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Dict for storing bounding boxes of each object.
        tracks = {
            'players':[],
            'referee':[],
            'ball':[]
        }
        
        # How we are storing the data -->
        # tracks = {
        #     'players': [
        #         {0:{"bbox":[0,0,0,0]}, 1:{"bbox":[0,0,0,0]}, 21:{"bbox":[0,0,0,0]}},    Frame 1
        #         {0:{"bbox":[0,0,0,0]}, 1:{"bbox":[0,0,0,0]}, 21:{"bbox":[0,0,0,0]}},    Frame 2
        #     ]
        # }
        
        # Getting the detections from YOLO
        detections = self.detect_frames(frames)
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names        # {0: person, 1: ball ...}
            cls_names_inv = {v:k for k,v in cls_names.items()}    # {person: 0, ball: 1 ...}
            
            # Convert the detections to supervision detection format by creating sv.Detections object instance.
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # Our model was confusing the goalkeeper with normal player.
            # Since we're not doing any specific analysis for goalkeeper,
            # We're gonna replace the goalkeeper with player.
            for obj_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[obj_idx] = cls_names_inv['player']
                    detection_supervision.data['class_name'][obj_idx] = 'player'
                    
            # Track objects - Adding tracking information to our detections.
            # Contains information for all the frames in the video.
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # Appending a dict for each frame that passes by.
            tracks['players'].append({})
            tracks['referee'].append({})
            tracks['ball'].append({})
            
            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()  # list of bboxes for each object in frame
                cls_id = frame_detection[3]         # class id of object corresponding to each bbox.
                track_id = frame_detection[4]       # tracking id of object corresponding to each bbox
                
                if cls_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox} # Adding bbox to dict
                    
                if cls_id == cls_names_inv['referee']:
                    tracks['referee'][frame_num][track_id] = {'bbox': bbox} # Adding bbox to dict
            
            # Since there is only one ball, no need for track_id. We can just use 1.
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox":bbox}
                    
        # Saving tracks dict.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            
        return tracks
    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = getCenterOfBbox(bbox)
        
        triangle = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ], dtype=np.int32)
        
        # Reshaping for polylines function's format
        triangle = triangle.reshape((-1, 1, 2))
        
        # Creating the triangle
        cv2.fillPoly(
            frame,
            [triangle],
            color=color,
            lineType=cv2.LINE_4
        )
        # Creating a border along the triangle
        cv2.polylines(
            frame,
            [triangle],
            isClosed=False,
            color=(0,0,0),
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        return frame
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])   # Bottom coordinate.
        x_center, _ = getCenterOfBbox(bbox)
        width = getBboxWidth(bbox)
        
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width//2)
        x2_rect = int(x_center + rectangle_width//2)
        y1_rect = int((y2 - rectangle_height//2) + 15)
        y2_rect = int((y2 + rectangle_height//2) + 15)
        
        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            )
            
            x1_text = x1_rect + 12  # Adding padding to the 
            if track_id > 99:
                x1_text -= 10   # Bigger number so shift to left.
                
            cv2.putText(
                frame,
                f'{track_id}',
                (int(x1_text), y1_rect+15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        
        return frame
        
    
    def draw_annotations(self, video_frames, tracks):
        """
        We want to draw a circle below the players, referees.
        Also want to track the ball with a small triangle.
        """
        
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()    # Creating a copy so original list remains intact.
            
            player_dict = tracks['players'][frame_num]  # Contains bboxes of all players in this frame
            referee_dict = tracks['referee'][frame_num] # Contains bboxes of all referees in the frame
            ball_dict = tracks['ball'][frame_num]       # Contains bbox of ball in this frame
            
            # Draw players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)
                
            # Draw referees
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))
                
            # Draw ball tracker
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (255, 255, 0))
                
            output_video_frames.append(frame)
            
        return output_video_frames