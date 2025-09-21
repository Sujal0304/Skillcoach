import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from video_processor import VideoProcessor
from pose_estimator import PoseEstimator
from boxing_analyzer import BoxingAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import time

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = BoxingAnalyzer()
if 'pose_estimator' not in st.session_state:
    st.session_state.pose_estimator = PoseEstimator()
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()

st.set_page_config(
    page_title="Boxing Motion Analysis",
    page_icon="🥊",
    layout="wide"
)

st.title("🥊 Boxing Motion Analysis - Side by Side Comparison")
st.markdown("Upload two boxing videos to compare techniques in real-time")

# Sidebar for analysis settings
st.sidebar.header("Analysis Settings")
frame_rate = st.sidebar.slider(
    "Processing Frame Rate (FPS)",
    min_value=5,
    max_value=15,
    value=8,
    help="Lower frame rate = faster processing"
)

sync_videos = st.sidebar.checkbox(
    "Sync Video Playback",
    value=True,
    help="Synchronize both videos for frame-by-frame comparison"
)

# Main content area - Two video uploads
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📹 Reference Video (Expert/Coach)")
    reference_file = st.file_uploader(
        "Upload reference boxing video",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video showing correct boxing technique",
        key="reference_video"
    )

with col2:
    st.header("📹 User Video (Your Technique)")
    user_file = st.file_uploader(
        "Upload your boxing video",
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of your boxing technique to analyze",
        key="user_video"
    )

# Process videos when both are uploaded
if reference_file is not None and user_file is not None:
    # Display upload status
    col1.success(f"Reference: {reference_file.name}")
    col2.success(f"User video: {user_file.name}")
    
    # Center the analysis button
    st.markdown("---")
    _, center_col, _ = st.columns([1, 1, 1])
    
    with center_col:
        if st.button("🚀 Compare Videos", type="primary"):
            with st.spinner("Processing both videos... This may take up to 2 minutes"):
                start_time = time.time()
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Save both files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ref_tmp:
                        ref_tmp.write(reference_file.read())
                        ref_video_path = ref_tmp.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as user_tmp:
                        user_tmp.write(user_file.read())
                        user_video_path = user_tmp.name
                    
                    # Step 1: Extract frames from both videos
                    status_text.text("Extracting frames from both videos...")
                    progress_bar.progress(0.2)
                    
                    ref_frames = st.session_state.video_processor.extract_frames(
                        ref_video_path, target_fps=frame_rate
                    )
                    user_frames = st.session_state.video_processor.extract_frames(
                        user_video_path, target_fps=frame_rate
                    )
                    
                    # Step 2: Pose estimation for both videos
                    status_text.text("Analyzing poses in both videos...")
                    progress_bar.progress(0.4)
                    
                    ref_poses = []
                    user_poses = []
                    
                    # Process reference video poses
                    for i, frame in enumerate(ref_frames):
                        pose = st.session_state.pose_estimator.estimate_pose(frame)
                        if pose is not None:
                            ref_poses.append(pose)
                        progress_bar.progress(0.4 + (i / (len(ref_frames) + len(user_frames))) * 0.4)
                    
                    # Process user video poses
                    for i, frame in enumerate(user_frames):
                        pose = st.session_state.pose_estimator.estimate_pose(frame)
                        if pose is not None:
                            user_poses.append(pose)
                        progress_bar.progress(0.4 + ((len(ref_frames) + i) / (len(ref_frames) + len(user_frames))) * 0.4)
                    
                    # Step 3: Compare videos
                    status_text.text("Comparing video techniques...")
                    progress_bar.progress(0.8)
                    
                    if ref_poses and user_poses:
                        comparison_result = st.session_state.analyzer.compare_videos(
                            ref_poses, user_poses, ref_frames, user_frames
                        )
                        
                        # Store results in session state
                        st.session_state.comparison_result = comparison_result
                        st.session_state.ref_frames = ref_frames
                        st.session_state.user_frames = user_frames
                        st.session_state.ref_poses = ref_poses
                        st.session_state.user_poses = user_poses
                        
                        progress_bar.progress(1.0)
                        processing_time = time.time() - start_time
                        status_text.text(f"Comparison complete in {processing_time:.1f} seconds!")
                        
                        st.rerun()
                    else:
                        st.error("No poses detected in one or both videos. Please ensure people are clearly visible.")
                
                except Exception as e:
                    st.error(f"Error processing videos: {str(e)}")
                finally:
                    # Clean up temporary files
                    if 'ref_video_path' in locals() and os.path.exists(ref_video_path):
                        os.unlink(ref_video_path)
                    if 'user_video_path' in locals() and os.path.exists(user_video_path):
                        os.unlink(user_video_path)

elif reference_file is not None or user_file is not None:
    st.info("📋 Please upload both videos to start comparison")
    if reference_file is None:
        st.warning("Missing: Reference video (expert/coach technique)")
    if user_file is None:
        st.warning("Missing: User video (your technique)")

# Display comparison results
if 'comparison_result' in st.session_state:
    st.markdown("---")
    st.header("📈 Comparison Results")
    
    result = st.session_state.comparison_result
    
    # Overall similarity score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        similarity_score = result.get('overall_similarity', 0)
        st.metric(
            label="Overall Technique Similarity",
            value=f"{similarity_score:.1f}%",
            delta=f"{similarity_score - 80:.1f}%" if similarity_score != 80 else None
        )
    
    # Side by side comparison
    st.subheader("🔍 Side-by-Side Frame Comparison")
    
    if 'ref_frames' in st.session_state and 'user_frames' in st.session_state:
        ref_frames = st.session_state.ref_frames
        user_frames = st.session_state.user_frames
        ref_poses = st.session_state.ref_poses
        user_poses = st.session_state.user_poses
        
        # Frame selector
        max_frames = min(len(ref_frames), len(user_frames))
        if max_frames > 0:
            frame_idx = st.slider(
                "Select Frame for Comparison",
                0,
                max_frames - 1,
                max_frames // 2,
                help="Drag to compare different frames"
            )
            
            # Display frames side by side
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🏆 Reference (Expert)")
                if frame_idx < len(ref_frames) and frame_idx < len(ref_poses):
                    ref_frame = ref_frames[frame_idx]
                    ref_pose = ref_poses[frame_idx]
                    # Draw reference in green (good)
                    annotated_ref = st.session_state.pose_estimator.draw_pose(ref_frame, ref_pose)
                    st.image(annotated_ref, caption="Reference Technique", use_column_width=True)
            
            with col2:
                st.subheader("📹 Your Technique")
                if frame_idx < len(user_frames) and frame_idx < len(user_poses):
                    user_frame = user_frames[frame_idx]
                    user_pose = user_poses[frame_idx]
                    
                    # Get differences for this frame to highlight problems
                    frame_differences = None
                    if 'differences' in result:
                        frame_differences = result['differences']
                    
                    # Draw user technique with highlighting based on differences
                    annotated_user = st.session_state.pose_estimator.draw_pose(
                        user_frame, user_pose, highlight_differences=frame_differences
                    )
                    st.image(annotated_user, caption="Your Technique", use_column_width=True)
                    
                    # Show frame-specific feedback
                    if frame_differences:
                        st.write("**Issues in this frame:**")
                        issue_count = 0
                        for body_part, diff_data in frame_differences.items():
                            if diff_data['difference'] > 15:  # Significant difference
                                if issue_count < 3:  # Limit to top 3 issues
                                    severity_color = "🔴" if diff_data['severity'] == "High" else "🟡"
                                    st.write(f"{severity_color} {body_part.replace('_', ' ').title()}: {diff_data['difference']:.1f}° difference")
                                    issue_count += 1
    
    # Detailed feedback with visual highlights
    st.subheader("💡 Areas for Improvement")
    
    if 'differences' in result:
        differences = result['differences']
        
        for body_part, diff_data in differences.items():
            if diff_data['difference'] > 15:  # Show significant differences
                with st.expander(f"⚠️ {body_part.replace('_', ' ').title()} - {diff_data['severity']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"• **Difference:** {diff_data['difference']:.1f}°")
                        st.write(f"• **Issue:** {diff_data['description']}")
                        st.write(f"• **Suggestion:** {diff_data['suggestion']}")
                    
                    with col2:
                        # Show severity level
                        severity = diff_data['severity']
                        if severity == "High":
                            st.error("Needs significant improvement")
                        elif severity == "Medium":
                            st.warning("Room for improvement")
                        else:
                            st.info("Minor adjustment needed")
    
    # Performance chart
    if 'joint_similarities' in result:
        st.subheader("📊 Joint-by-Joint Comparison")
        similarities = result['joint_similarities']
        
        joints = list(similarities.keys())
        scores = list(similarities.values())
        
        fig = px.bar(
            x=joints,
            y=scores,
            title="Similarity Score by Body Part",
            labels={'x': 'Body Parts', 'y': 'Similarity (%)'},
            color=scores,
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Technical summary
    with st.expander("📄 Technical Analysis Summary"):
        st.json({
            "total_frames_compared": result.get('frames_compared', 0),
            "processing_fps": frame_rate,
            "sync_enabled": sync_videos,
            "analysis_method": "video_to_video_comparison",
            "reference_video": reference_file.name if reference_file else "N/A",
            "user_video": user_file.name if user_file else "N/A"
        })
else:
    # Default instruction screen
    st.markdown("---")
    st.info("📋 Upload both videos above to begin side-by-side comparison analysis")
    
    # Instructions
    st.subheader("📝 How to Use")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **1. Upload Reference Video**
        • Expert or coach demonstration
        • Clear view of boxing technique
        • Good lighting and camera angle
        """)
    
    with col2:
        st.markdown("""
        **2. Upload Your Video**
        • Your boxing technique attempt
        • Similar camera angle as reference
        • Same technique being performed
        """)

# Footer
st.markdown("---")
st.markdown("🥊 **Boxing Motion Analysis System** - Real-time side-by-side video comparison")