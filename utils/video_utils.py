# Read in the video and save the video
import cv2

def read_video(video_path):
    """Takes in a video path and returns a list of frames for the video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):
    """Takes in the frames and saves the video."""
    codec = cv2.VideoWriter_fourcc(*'XVID')   
    out = cv2.VideoWriter(output_video_path, codec, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()