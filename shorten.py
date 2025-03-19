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



def get_frame_count(video_path, interval=0.1):
    """
    Calculate the total number of frames that will be processed given the interval.
    
    Args:
        video_path: Path to the video file
        interval: Time interval between frames in seconds
        
    Returns:
        total_processed_frames: Number of frames that will be processed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(round(interval * fps)))  # Convert seconds to frame count
        return total_frames // frame_interval + 1
    finally:
        cap.release()

def extract_frames(video_path, interval=0.1, batch_size=16):
    """
    Generator that yields batches of (timestamps, frames) from the video at given interval.
    
    Args:
        video_path: Path to the video file
        interval: Time interval between frames in seconds
        batch_size: Number of frames to yield at once
        
    Yields:
        tuple: (timestamps, frames) where each is a list containing batch_size items
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(round(interval * fps)))  # Convert seconds to frame count

    frame_idx = 0
    timestamp_batch = []
    frame_batch = []
    
    try:
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_batch.append(frame_idx / fps)
            frame_batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # When batch is full, yield the batch
            if len(frame_batch) >= batch_size:
                yield timestamp_batch, frame_batch
                timestamp_batch = []
                frame_batch = []
                
            frame_idx += frame_interval
            
        # Don't forget the last batch if it's not empty
        if frame_batch:
            yield timestamp_batch, frame_batch
    finally:
        cap.release()

class VideoEmbeddingModel:
    """
    A class that handles frame embedding extraction for video analysis.
    Encapsulates model initialization, preprocessing, and embedding computation.
    """
    def __init__(self, embedding_method="ResNet18"):
        """Initialize the model and preprocessing pipeline."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if embedding_method == "ResNet18":
            net = models.resnet18(pretrained=True)
        elif embedding_method == "ShuffleNet":
            net = models.shufflenet_v2_x0_5(pretrained=True)

        self.model = torch.nn.Sequential(*(list(net.children())[:-1])).to(device).eval()
        self.device = device
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def compute_embeddings(self, frames):
        """
        Compute image embeddings for a batch of frames.
        
        Args:
            frames: List of RGB frames (numpy arrays)
            
        Returns:
            numpy array of embeddings with shape (len(frames), embedding_dim)
        """
        with torch.no_grad():
            input_tensors = torch.stack([self.preprocess(Image.fromarray(frame)) for frame in frames]).to(self.device)
            embeddings = self.model(input_tensors).cpu().numpy()
        
        return embeddings.reshape(len(frames), -1)

def detect_interesting_segments(embeddings, timestamps, target_duration=20.0, segment_padding=0.5, 
                               kernel_size=5, distance=9, graph_output_file="segments_graph.png"):
    """
    Detect interesting segments based on changes in frame embeddings.
    This function computes the L2 norm difference between successive embeddings,
    smooths the signal, and selects peaks.
    
    Args:
        embeddings: Frame embeddings
        timestamps: Corresponding timestamps for each frame
        target_duration: Target duration of the selected segments in seconds
        segment_padding: Amount of padding around each peak in seconds
        kernel_size: Size of the kernel for median filtering
        distance: Minimum distance between peaks
        graph_output_file: File to save the visualization graph
        
    Returns:
        segments: a list of tuples (start, end) of selected segments.
    """
    # Compute difference between successive embeddings
    diffs = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    
    # Smooth the differences to reduce noise (using a median filter)
    smooth_diffs = scipy.signal.medfilt(diffs, kernel_size=kernel_size)
    
    # Find peaks (with a minimum distance between peaks)
    peaks, _ = scipy.signal.find_peaks(smooth_diffs, distance=distance)
    
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

def assemble_final_video_streaming(video_path, segments, output_path, vertical=False):
    """
    Assemble final video using streaming approach, only loading required segments.
    """
    with VideoFileClip(video_path) as clip:
        subclips = [clip.subclip(start, end) for start, end in segments]
        final_clip = concatenate_videoclips(subclips, method="compose")
        
        if vertical:
            final_clip = final_clip.resize(newsize=(clip.h, clip.w))
        else:
            final_clip = final_clip.resize((clip.w, clip.h))
        
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatically shorten a video by selecting interesting moments.")
    parser.add_argument('--video', type=str, required=True, help="Path to the input video")
    parser.add_argument('--output', type=str, default='output.mp4', help="Path for the output video")
    parser.add_argument('--duration', type=float, default=20.0, help="Target duration (in seconds) of the output video")
    parser.add_argument('--graph', type=str, default="segments_graph.png", help="Filename for the output graph")
    parser.add_argument('--vertical', action='store_true', help="Indicate that the video was filmed vertically")
    parser.add_argument('--kernel-size', type=int, default=5, help="Size of the kernel for median filtering")
    parser.add_argument('--distance', type=int, default=9, help="Minimum distance between peaks for peak detection")
    parser.add_argument('--interval', type=float, default=3.0, help="Frame sampling interval in seconds")
    args = parser.parse_args()

    print("Loading model...")
    embedding_model = VideoEmbeddingModel()
    
    print("Processing video frames...")
    embeddings_data = []
    
    # Get total number of frames for progress tracking
    total_frames = get_frame_count(args.video, interval=args.interval)
    
    # Create a progress bar for the entire processing
    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        for timestamps, frames in extract_frames(args.video, interval=args.interval, batch_size=16):
            # Process the entire batch at once
            embeddings = embedding_model.compute_embeddings(frames)
            
            # Store the timestamp and embedding for each frame
            for ts, emb in zip(timestamps, embeddings):
                embeddings_data.append((ts, emb))
                
            # Update progress bar with the batch size
            pbar.update(len(frames))
    
    print("Detecting interesting segments and generating graph...")
    timestamps, embeddings = zip(*embeddings_data)
    segments = detect_interesting_segments(
        np.array(embeddings),
        timestamps,
        target_duration=args.duration,
        kernel_size=args.kernel_size,
        distance=args.distance,
        graph_output_file=args.graph
    )
    
    print("Assembling the final video...")
    assemble_final_video_streaming(args.video, segments, args.output, vertical=args.vertical)
    
    print("Done! The output video is saved at:", args.output)
