import cv2
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import models
from moviepy.editor import VideoFileClip, concatenate_videoclips
from PIL import Image
import scipy.signal
import argparse
import matplotlib.pyplot as plt

# def extract_frames(video_path, interval=0.1):
#     """
#     Extract frames from the video at the given interval (in seconds).
#     Returns a list of timestamps and corresponding RGB frames.
#     """
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     timestamps = []
#     current_time = 0.0
#     while True:
#         cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         # Convert from BGR (OpenCV default) to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame_rgb)
#         timestamps.append(current_time)
#         current_time += interval
#     cap.release()
#     return timestamps, frames

def extract_frames(video_path, interval=0.1):
    """
    Extract frames from the video at the given interval (in seconds).
    Returns a list of timestamps and corresponding RGB frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(interval * fps)))  # Convert seconds to frame count

    frames, timestamps = [], []
    frame_idx = 0

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamps.append(frame_idx / fps)
            frame_idx += frame_interval
            pbar.update(frame_interval)

    cap.release()
    return timestamps, frames

def compute_embeddings(frames, model, preprocess):
    """
    Compute image embeddings for a batch of frames using the provided model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    with torch.no_grad():
        input_tensors = torch.stack([preprocess(Image.fromarray(frame)) for frame in frames]).to(device)
        embeddings = model(input_tensors).cpu().numpy()
    
    return embeddings.reshape(len(frames), -1)

def detect_interesting_segments(embeddings, timestamps, target_duration=20.0, segment_padding=0.5, graph_output_file="segments_graph.png"):
    """
    Detect interesting segments based on changes in frame embeddings.
    This function computes the L2 norm difference between successive embeddings,
    smooths the signal, and selects peaks.
    
    It also generates a graph of the smoothed differences, marks all detected peaks,
    and highlights the segments chosen (with padding). The graph is saved to the given file.
    
    Returns:
        segments: a list of tuples (start, end) of selected segments.
    """
    # Compute difference between successive embeddings
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    
    # Smooth the differences to reduce noise (using a median filter)
    smooth_diffs = scipy.signal.medfilt(diffs, kernel_size=5)
    
    # Find peaks (with a minimum distance between peaks)
    peaks, _ = scipy.signal.find_peaks(smooth_diffs, distance=9)
    
    # Sort peaks by their novelty score (largest difference first)
    sorted_peaks = sorted(peaks, key=lambda idx: smooth_diffs[idx], reverse=True)
    
    segments = []
    total_duration = 0.0
    for idx in tqdm(sorted_peaks, 
                    desc="Iterating over interesting peaks",
                    total=round(target_duration / (segment_padding*2))):
        if total_duration >= target_duration:
            break
        # Define a segment around the peak with some padding
        start = max(0, timestamps[idx] - segment_padding)
        # Use next timestamp if available
        # if idx + 1 < len(timestamps):
        #     end = timestamps[idx+1] + segment_padding
        # else:
        end = timestamps[idx] + segment_padding
        segment_duration = end - start
        segments.append((start, end))
        total_duration += segment_duration
    
    # Sort segments in chronological order to keep the story intact
    segments.sort(key=lambda seg: seg[0])
    
    # Plotting the graph
    x_axis = np.array(timestamps[1:])  # diffs corresponds to timestamps[1:]
    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, smooth_diffs, label="Smoothed Difference")
    plt.scatter(x_axis[peaks], smooth_diffs[peaks], color='red', label="Detected Peaks")
    
    # Highlight the selected segments with a shaded region
    for (start, end) in segments:
        plt.axvspan(start, end, color='green', alpha=0.3)
    
    plt.title("Smoothed Embedding Differences with Selected Segments")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference Magnitude")
    plt.legend()
    plt.savefig(graph_output_file)
    plt.close()
    
    print(f"Graph saved as: {graph_output_file}")
    
    return segments

def assemble_final_video(video_path, segments, output_path, vertical=False):
    """
    Extract segments from the original video and concatenate them.
    If the video is filmed vertical, rotate and resize to portrait dimensions.
    Otherwise, preserve the original (landscape) dimensions.
    """
    clip = VideoFileClip(video_path)
    original_width, original_height = clip.size

    subclips = [clip.subclip(start, end) for start, end in segments]
    # Use `method="compose"` to maintain aspect ratios
    final_clip = concatenate_videoclips(subclips, method="compose")
    
    if vertical:
        # Assume the intended portrait resolution is 1080x1920.
        # Rotate by 90 degrees and then resize.
        final_clip = final_clip.resize(newsize=(original_height, original_width))
    else:
        # Preserve original landscape dimensions.
        final_clip = final_clip.resize((original_width, original_height))
    
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically shorten a video by selecting interesting moments.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video")
    parser.add_argument('--output', type=str, default='output.mp4', help="Path for the output video")
    parser.add_argument('--duration', type=float, default=20.0, help="Target duration (in seconds) of the output video")
    parser.add_argument('--graph', type=str, default="segments_graph.png", help="Filename for the output graph")
    parser.add_argument('--vertical', action='store_true', help="Indicate that the video was filmed vertically")
    args = parser.parse_args()

    print("Extracting frames...")
    timestamps, frames = extract_frames(args.video, interval=3)
    print("Loading model...")
    # Load a pretrained ResNet18 and remove the final classification layer to obtain embeddings
    resnet = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    print("Computing embeddings for frames...")
    embeddings = compute_embeddings(frames, model, preprocess)
    
    print("Detecting interesting segments and generating graph...")
    segments = detect_interesting_segments(embeddings, 
                                           timestamps, 
                                           target_duration=args.duration, 
                                           graph_output_file=args.graph)
    
    print("Assembling the final video...")
    assemble_final_video(args.video, segments, args.output, vertical=args.vertical)
    
    print("Done! The output video is saved at:", args.output)
