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
import imagehash


def extract_frames(video_path, interval=0.1, batch_size=16,
                   filter_subsequent_duplicates=True):
    """
    Generator that yields batches of (timestamps, frames, hashes, seconds_processed) from the video.
    
    Args:
        video_path: Path to the video file
        interval: Time interval between frames in seconds
        batch_size: Number of frames to yield at once
        filter_subsequent_duplicates: If True, skip frames that are identical to previous
        
    Yields:
        tuple: (timestamps, frames, hashes, seconds_processed) where:
            - timestamps: list of timestamps for each frame
            - frames: list of RGB frames
            - hashes: list of image hash objects
            - seconds_processed: cumulative count of seconds processed
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
    hash_batch = []
    previous_hash = None
    seconds_processed = 0
    
    try:
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_hash = get_frame_hash(current_frame_rgb)
            seconds_processed += interval
            
            # Check if we should filter duplicates
            include_frame = True
            if filter_subsequent_duplicates and previous_hash is not None:
                if current_hash == previous_hash:
                    print(f"Skipping duplicate frame at {frame_idx}")
                    include_frame = False
                else:
                    previous_hash = current_hash
            elif filter_subsequent_duplicates:
                # First frame, set the hash
                previous_hash = current_hash
                
            if include_frame:
                timestamp_batch.append(frame_idx / fps)
                frame_batch.append(current_frame_rgb)
                hash_batch.append(current_hash)
                
                # When batch is full, yield the batch
                if len(frame_batch) >= batch_size:
                    yield timestamp_batch, frame_batch, hash_batch, seconds_processed
                    timestamp_batch = []
                    frame_batch = []
                    hash_batch = []
                
            frame_idx += frame_interval
            
        # Don't forget the last batch if it's not empty
        if frame_batch:
            yield timestamp_batch, frame_batch, hash_batch, seconds_processed
    finally:
        cap.release()


def get_frame_hash(binary_frame):
    """
    Compute a hash value for a frame using the imagehash library.
    
    Args:
        binary_frame: A binary frame (numpy array)
        
    Returns:
        hash: A hash value for the frame
    """
    return imagehash.dhash(Image.fromarray(binary_frame))


class VideoHashModel:
    """
    A class that handles frame hash computation for video analysis.
    Designed as a drop-in replacement for VideoEmbeddingModel.
    """
    def __init__(self, hash_method="dhash", hash_size=8):
        """Initialize the hash computation settings."""
        self.hash_method = hash_method
        self.hash_size = hash_size
        
    def compute_embeddings(self, frames):
        """
        Compute hash-based 'embeddings' for a batch of frames.
        Compatible with VideoEmbeddingModel's interface.
        
        Args:
            frames: List of RGB frames (numpy arrays)
            
        Returns:
            numpy array of hash values converted to vectors
        """
        # Compute hash for each frame and convert to vector representation
        hash_vectors = []
        for frame in frames:
            frame_hash = self._compute_hash(frame)
            # Convert hash to a binary vector for numerical operations
            hash_vector = np.array(list(frame_hash.hash.flatten()))
            hash_vectors.append(hash_vector)
        
        return np.array(hash_vectors)
    
    
    def _compute_hash(self, frame):
        """
        Compute image hash for a frame.
        
        Args:
            frame: RGB frame (numpy array)
            
        Returns:
            hash: An imagehash object
        """
        img = Image.fromarray(frame)
        if self.hash_method == "dhash":
            return imagehash.dhash(img, hash_size=self.hash_size)
        elif self.hash_method == "phash":
            return imagehash.phash(img, hash_size=self.hash_size)
        elif self.hash_method == "average_hash":
            return imagehash.average_hash(img, hash_size=self.hash_size)
        else:
            return imagehash.dhash(img, hash_size=self.hash_size)
    
    @classmethod
    def compute_diffs(cls, embeddings):
        """
        Compute the difference between successive hash vectors.
        Compatible with VideoEmbeddingModel's interface.
        
        Args:
            embeddings: Hash vectors as returned by compute_embeddings
            
        Returns:
            numpy array of differences between consecutive hash vectors
        """
        # For hash vectors (binary), Hamming distance is suitable
        # (equivalent to counting the number of different bits)
        return np.sum(np.abs(np.diff(embeddings, axis=0)), axis=1)


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
    
    @classmethod
    def compute_diffs(cls, embeddings):
        """
        Compute the difference between successive embeddings.
        """
        return np.linalg.norm(np.diff(embeddings, axis=0), axis=1)



def detect_interesting_segments(diffs, timestamps, target_duration=20.0, segment_padding=0.5, 
                               kernel_size=5, distance=9, graph_output_file="segments_graph.png",
                               inverse=False):
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
    
    # Smooth the differences to reduce noise (using a median filter)
    if inverse:
        smooth_diffs = scipy.signal.medfilt(diffs, kernel_size=kernel_size)
    else:
        smooth_diffs = scipy.signal.medfilt(-diffs, kernel_size=kernel_size)

    
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
    Assumes segments is a list of dictionaries: [{'start_time': s, 'end_time': e}, ...]
    """
    # Sort segments by start time to ensure chronological order
    sorted_segments = sorted(segments, key=lambda x: x['start_time'])
    
    with VideoFileClip(video_path) as clip:
        # Extract start and end times from the sorted dictionaries
        subclips = []
        for seg in sorted_segments:
            subclips.append(clip.subclip(seg['start_time'], seg['end_time']))
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
    parser.add_argument('--duration', type=float, default=20.0, 
                        help="Target duration (in seconds) of the output video")
    parser.add_argument('--graph', type=str, default="segments_graph.png", 
                        help="Filename for the output graph")
    parser.add_argument('--vertical', action='store_true', 
                        help="Indicate that the video was filmed vertically")
    parser.add_argument('--kernel-size', type=int, default=5, 
                        help="Size of the kernel for median filtering")
    parser.add_argument('--distance', type=int, default=9, 
                        help="Minimum distance between peaks for peak detection")
    parser.add_argument('--interval', type=float, default=3.0, 
                        help="Frame sampling interval in seconds")
    parser.add_argument('--use-hash', action='store_true',
                        help="Use hash-based processing instead of deep learning embeddings")
    args = parser.parse_args()

    print("Processing video...")
    if args.use_hash:
        print("Using hash-based processing...")
        model = VideoHashModel(hash_method="dhash", hash_size=8)
        embeddings_data = []
        for timestamps_batch, frames_batch, hashes_batch, frame_count in extract_frames(
            args.video, 
            interval=args.interval,
            filter_subsequent_duplicates=True
        ):
            # Use pre-computed hashes instead of recomputing from frames
            embeddings_batch = model.compute_embeddings_from_hashes(hashes_batch)
            for ts, emb in zip(timestamps_batch, embeddings_batch):
                embeddings_data.append((ts, emb))
                
        timestamps, embeddings = zip(*embeddings_data)
        embeddings = np.array(embeddings)
        diffs = model.compute_diffs(embeddings)
    else:
        print("Loading embedding model...")
        model = VideoEmbeddingModel()
        
        # Get total number of frames using a quick read of the video
        cap = cv2.VideoCapture(args.video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        interval_frames = max(1, int(round(args.interval * fps)))
        estimated_frames = total_frames // interval_frames
        
        # Process frames using embedding model
        embeddings_data = []
        with tqdm(total=estimated_frames, desc="Processing frames") as pbar:
            for timestamps_batch, frames_batch, hashes_batch, frame_count in extract_frames(
                args.video, 
                interval=args.interval, 
                batch_size=16
            ):
                # Process the entire batch at once
                embeddings_batch = model.compute_embeddings(frames_batch)
                
                # Store the timestamp and embedding for each frame
                for ts, emb in zip(timestamps_batch, embeddings_batch):
                    embeddings_data.append((ts, emb))
                    
                # Update progress bar with the batch size
                pbar.update(len(frames_batch))
                
        timestamps, embeddings = zip(*embeddings_data)
        embeddings = np.array(embeddings)
        diffs = model.compute_diffs(embeddings)
    
    print("Detecting interesting segments and generating graph...")
    segments = detect_interesting_segments(
        diffs,
        timestamps,
        target_duration=args.duration,
        kernel_size=args.kernel_size,
        distance=args.distance,
        graph_output_file=args.graph
    )
    
    print("Assembling the final video...")
    assemble_final_video_streaming(args.video, segments, args.output, vertical=args.vertical)
    
    print("Done! The output video is saved at:", args.output)
