import streamlit as st
import os
import tempfile
from shorten import (
    extract_frames,
    VideoEmbeddingModel,
    detect_interesting_segments,
    assemble_final_video_streaming,
    get_frame_count,
)
import matplotlib.pyplot as plt
import time
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips

st.set_page_config(page_title="Video Shortener", layout="wide")

# Initialize session state for tracking progress
if 'steps' not in st.session_state:
    st.session_state.steps = {
        'upload': {'status': 'waiting', 'complete': False},
        'extract_frames': {'status': 'waiting', 'complete': False},
        'compute_embeddings': {'status': 'waiting', 'complete': False},
        'find_peaks': {'status': 'waiting', 'complete': False},
        'generate_video': {'status': 'waiting', 'complete': False}
    }
if 'progress' not in st.session_state:
    st.session_state.progress = {}
if 'embeddings_data' not in st.session_state:
    st.session_state.embeddings_data = []  # Will store (timestamp, embedding) pairs
if 'preview_frames' not in st.session_state:
    st.session_state.preview_frames = {}  # Will store {timestamp: frame} for previews
if 'timestamps' not in st.session_state:
    st.session_state.timestamps = None
if 'frames' not in st.session_state:
    st.session_state.frames = None
if 'segments' not in st.session_state:
    st.session_state.segments = None
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'peaks_detected' not in st.session_state:
    st.session_state.peaks_detected = False
if 'graph_file' not in st.session_state:
    # Create a temporary file for the graph
    temp_graph_dir = tempfile.mkdtemp()
    st.session_state.graph_file = os.path.join(temp_graph_dir, "segments_graph.png")

# Sidebar for progress tracking
st.sidebar.title("Processing Status")

# Create placeholder for status updates in the sidebar
if 'status_container' not in st.session_state:
    st.session_state.status_container = st.sidebar.empty()

# Display status of each step in the sidebar
def update_sidebar():
    steps_list = ['upload', 'extract_frames', 'compute_embeddings', 'find_peaks', 'generate_video']
    step_names = {
        'upload': 'Upload Video',
        'extract_frames': 'Extract Frames',
        'compute_embeddings': 'Compute Embeddings',
        'find_peaks': 'Find Interesting Peaks',
        'generate_video': 'Generate Shortened Video'
    }
    
    # Use a single container for all status updates
    with st.session_state.status_container.container():
        for step in steps_list:
            status = st.session_state.steps[step]['status']
            complete = st.session_state.steps[step]['complete']
            
            # Create a container for each step
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                # Show checkmark, spinner, or empty based on status
                if complete:
                    col1.success("✓")
                elif status == 'in_progress':
                    col1.info("⋯")
                else:
                    col1.empty()
                    
                # Show step name and status
                if status == 'waiting':
                    col2.text(f"{step_names[step]}: Waiting")
                elif status == 'in_progress':
                    col2.text(f"{step_names[step]}: In Progress")
                elif status == 'complete':
                    col2.text(f"{step_names[step]}: Complete")
                
                # Show progress bar if in progress
                if status == 'in_progress' and step in st.session_state.get('progress', {}):
                    st.progress(st.session_state.progress[step])
        
        st.markdown("---")

# Main content
st.title("Smart Video Shortener")
st.write("Upload a video and create a shorter version with the most interesting moments.")

# Replace the file uploader section with directory listing
video_dir = "videos"  # Directory containing the videos
if not os.path.exists(video_dir):
    st.error(f"Please create a '{video_dir}' directory and add your videos there.")
else:
    # List all video files in the directory
    video_files = [f for f in os.listdir(video_dir) 
                  if f.lower().endswith(('.mp4', '.mov', '.avi'))]
    
    if not video_files:
        st.warning(f"No video files found in '{video_dir}' directory. Please add some videos.")
    else:
        # Create a dropdown to select the video
        selected_video = st.selectbox(
            "Select a video to process",
            video_files,
            format_func=lambda x: os.path.splitext(x)[0]  # Show filename without extension
        )
        
        if selected_video:
            video_path = os.path.join(video_dir, selected_video)
            st.session_state.video_path = video_path
            
            # Update upload status
            if not st.session_state.steps['upload']['complete']:
                st.session_state.steps['upload']['status'] = 'complete'
                st.session_state.steps['upload']['complete'] = True
                # Reset subsequent steps
                for step in ['extract_frames', 'compute_embeddings', 'find_peaks', 'generate_video']:
                    st.session_state.steps[step]['status'] = 'waiting'
                    st.session_state.steps[step]['complete'] = False
                st.session_state.processing_complete = False
                st.session_state.peaks_detected = False

# Process video button and parameters
col1, col2 = st.columns(2)
with col1:
    if st.session_state.video_path:
        interval = st.slider("Frame sampling interval (seconds)", 0.1, 5.0, 1.0, step=0.1)
        segment_padding = st.slider("Segment padding (seconds)", 0.5, 3.0, 1.0, step=0.1)
        
        embedding_method = st.selectbox("Embedding method", ["ResNet18", "ShuffleNet"])
        embedding_model = VideoEmbeddingModel(embedding_method)
        if st.button("Process Video"):
            # Reset state
            st.session_state.processing_complete = False
            st.session_state.peaks_detected = False
            st.session_state.embeddings_data = []
            st.session_state.preview_frames = {}
            
            # Use the original video path directly instead of copying to temp location
            # No need to create temporary files for the source video
            
            # Process frames with streaming
            st.session_state.steps['extract_frames']['status'] = 'in_progress'
            st.session_state.steps['compute_embeddings']['status'] = 'in_progress'
            update_sidebar()
            
            with st.spinner("Processing video frames..."):
                try:
                    # Get total frames for progress tracking
                    # total_frames = get_frame_count(st.session_state.video_path, interval=interval)
                    total_frames = 10_000
                    # Create a Streamlit progress bar
                    progress_text = "Processing video frames..."
                    progress_bar = st.progress(0)
                    
                    # Track processed frames
                    processed_frames = 0
                    
                    # Process frames in batches
                    for timestamps, frames in extract_frames(st.session_state.video_path, interval=interval):
                        embeddings = embedding_model.compute_embeddings(frames)
                        st.session_state.embeddings_data.extend(zip(timestamps, embeddings))
                        st.session_state.preview_frames.update(dict(zip(timestamps, frames)))
                        
                        # Update progress
                        processed_frames += len(frames)
                        progress_bar.progress(min(1.0, processed_frames / total_frames))
                    
                    # Complete the progress bar
                    progress_bar.progress(1.0)
                    
                    st.session_state.steps['extract_frames']['status'] = 'complete'
                    st.session_state.steps['extract_frames']['complete'] = True
                    st.session_state.steps['compute_embeddings']['status'] = 'complete'
                    st.session_state.steps['compute_embeddings']['complete'] = True
                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    st.session_state.steps['extract_frames']['status'] = 'error'
                    st.session_state.steps['compute_embeddings']['status'] = 'error'
            
            st.session_state.processing_complete = True
            st.success("Video processed successfully! You can now detect interesting peaks.")
            update_sidebar()

# Find interesting peaks section
if st.session_state.processing_complete:
    st.header("Find Interesting Peaks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_duration = st.slider("Target video duration (seconds)", 5, 60, 120, step=1)
        with st.expander("Advanced Settings"):
            kernel_size = st.slider("Kernel size for smoothing", 3, 15, 5, step=2)
            distance = st.slider("Minimum distance between peaks", 3, 20, 9)
        
        if st.button("Find Interesting Moments"):
            if st.session_state.embeddings_data:
                timestamps, embeddings = zip(*st.session_state.embeddings_data)
                segments = detect_interesting_segments(
                    np.array(embeddings),
                    timestamps,
                    target_duration=target_duration,
                    segment_padding=segment_padding,
                    kernel_size=kernel_size,
                    distance=distance,
                    graph_output_file=st.session_state.graph_file
                )
                
                # Assuming detect_interesting_segments returns segments with scores
                # If it doesn't currently return scores, you'll need to modify that function
                
                # Create a comprehensive frame data dictionary with all metadata
                frame_data = {}
                # segment_scores = {}
                
                # Process each segment with its importance score
                # If your detect_interesting_segments doesn't return scores,
                # you'll need to modify it to return (start, end, score) tuples
                for i, (start, end) in enumerate(segments):                    
                    # Store frames with segment metadata
                    relevant_timestamps = [t for t in st.session_state.preview_frames.keys()
                                        if start <= t <= end]
                    for t in relevant_timestamps:
                        frame_data[t] = {
                            'frame': st.session_state.preview_frames[t],
                            'timestamp': t,
                            'formatted_time': time.strftime('%M:%S', time.gmtime(t)),
                            'segment': (start, end),
                            # 'score': score
                        }
                
                st.session_state.frame_data = frame_data
                # st.session_state.segment_scores = segment_scores
                st.session_state.segments = segments
                st.session_state.steps['find_peaks']['status'] = 'complete'
                st.session_state.steps['find_peaks']['complete'] = True
                st.session_state.peaks_detected = True
                update_sidebar()
                
                st.success("Interesting moments detected! You can now select video clips.")
    
    # Display the segments graph if available
    if st.session_state.peaks_detected and os.path.exists(st.session_state.graph_file):
        with col2:
            st.image(st.session_state.graph_file, caption="Segments Analysis Graph")

# Display and select clips section
if st.session_state.peaks_detected:
    st.header("Select Video Clips")
    
    # Initialize selected_segments in session state if not exists
    if 'selected_segments' not in st.session_state:
        st.session_state.selected_segments = None
    
    if st.session_state.segments is not None:
        # Add filtering and sorting options
        col1, col2 = st.columns(2)
        
        # Filter segments by score and sort them
        filtered_segments = [(start, end) for (start, end) in st.session_state.segments]
        

        sorted_segments = sorted(filtered_segments, key=lambda x: x[0])
        
        # Create checkboxes for each segment
        selected_segments = []
        
        for c, (start_time, end_time) in enumerate(sorted_segments):
            # Format times to be more readable
            start_formatted = time.strftime('%M:%S', time.gmtime(start_time))
            end_formatted = time.strftime('%M:%S', time.gmtime(end_time))
            duration = end_time - start_time
            # score = st.session_state.segment_scores.get((start_time, end_time), 0)
            
            # Create a unique key for each checkbox
            checkbox_key = f"segment_{c}_{start_time}_{end_time}"
            
            # Create a container for each segment
            with st.container():
                col1, col2, col3 = st.columns([1, 3, 5])
                
                with col1:
                    if st.checkbox("Include", value=True, key=checkbox_key):
                        selected_segments.append((start_time, end_time))
                    st.write(f"Clip {c+1}")
                
                with col2:
                    st.write(f"{start_formatted} - {end_formatted}")
                    st.write(f"Duration: {duration:.1f}s")
                    # st.write(f"Score: {score:.2f}")
                
                with col3:
                    # Get frames within this segment
                    segment_timestamps = [t for t in st.session_state.frame_data.keys()
                                         if start_time <= t <= end_time]
                    
                    if segment_timestamps:
                        # Show the middle frame as a preview
                        middle_timestamp = segment_timestamps[len(segment_timestamps)//2]
                        frame_info = st.session_state.frame_data[middle_timestamp]
                        st.image(
                            frame_info['frame'], 
                            caption=f"Preview at {frame_info['formatted_time']}",
                            width=200
                        )
                
                # Add a separator between clips
                st.markdown("---")
        
        st.session_state.selected_segments = selected_segments
        
        total_duration = sum(end - start for start, end in selected_segments)
        st.info(f"Total selected duration: {total_duration:.1f} seconds")

# Generate shortened video section
if st.session_state.peaks_detected and st.session_state.selected_segments:
    st.header("Generate Shortened Video")
    
    vertical = st.checkbox("Video is vertical", value=False)
    
    if st.button("Generate Shortened Video"):
        if st.session_state.selected_segments:  # Changed from segments to selected_segments
            # Update generate video status
            st.session_state.steps['generate_video']['status'] = 'in_progress'
            st.session_state.progress['generate_video'] = 0
            update_sidebar()
            
            # Create output path
            output_dir = tempfile.mkdtemp()
            timestamp = int(time.time())
            output_path = os.path.join(output_dir, f"shortened_{timestamp}.mp4")
            
            # Assemble final video using selected segments
            with st.spinner("Assembling the final video..."):
                st.session_state.progress['generate_video'] = 20
                update_sidebar()
                
                assemble_final_video_streaming(
                    st.session_state.video_path,
                    st.session_state.selected_segments,  # Changed from segments to selected_segments
                    output_path,
                    vertical=vertical
                )
                
                st.session_state.progress['generate_video'] = 100
                update_sidebar()
            
            # Update status
            st.session_state.steps['generate_video']['status'] = 'complete'
            st.session_state.steps['generate_video']['complete'] = True
            update_sidebar()
            
            # Provide download link
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Shortened Video",
                    data=file,
                    file_name=f"shortened_{timestamp}.mp4",
                    mime="video/mp4"
                )
            
            # Display the video
            st.video(output_path)

# Show instructions if no file is uploaded
if not st.session_state.video_path:
    st.info("Please select a video to begin.")
    
    st.markdown("""
    ### How to use this app:
    1. Select a video from the directory
    2. Click "Process Video" to extract frames and compute embeddings
    3. Adjust parameters and find interesting moments
    4. Select video clips and generate the shortened video
    
    The app will keep video embeddings in memory, so you can try different parameters
    without having to reprocess the entire video.
    """)

# Update sidebar at the end
update_sidebar() 