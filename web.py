import streamlit as st
import tempfile
from video_processor import VideoProcessor
from pose_estimator import PoseEstimator
from boxing_analyzer import BoxingAnalyzer
from reference_poses import ReferenceBoxingPoses
from utils import PerformanceMonitor, format_feedback_message, create_accuracy_visualization

st.set_page_config(page_title="Boxing Skill Pose Comparator", layout="wide")

#---- Sidebar UI ----#
st.sidebar.header("Upload Videos")
user_video_file = st.sidebar.file_uploader("User Boxing Video", type=["mp4", "mov", "avi"])
ref_video_file = st.sidebar.file_uploader("Reference Athlete Video", type=["mp4", "mov", "avi"])

techniques = ReferenceBoxingPoses().get_all_techniques()
selected_technique = st.sidebar.selectbox("Choose boxing technique to compare", techniques)

if user_video_file and ref_video_file:
    st.info("Processing videos. This may take 1-2 minutes, depending on video length.")

    #---- Save uploaded videos to temporary files ----#
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as user_tmp:
        user_tmp.write(user_video_file.read())
        user_video_path = user_tmp.name
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ref_tmp:
        ref_tmp.write(ref_video_file.read())
        ref_video_path = ref_tmp.name

    #---- Initialize Utilities ----#
    video_processor = VideoProcessor()
    pose_estimator = PoseEstimator()
    boxing_analyzer = BoxingAnalyzer()
    perf_monitor = PerformanceMonitor()

    #---- Extract frames ----#
    user_frames = video_processor.extract_frames(user_video_path, target_fps=10)
    ref_frames  = video_processor.extract_frames(ref_video_path, target_fps=10)

    #---- Estimate poses ----#
    user_poses = [pose_estimator.estimate_pose(frame) for frame in user_frames]
    ref_poses  = [pose_estimator.estimate_pose(frame) for frame in ref_frames]

    #---- Analyze user technique against reference ----#
    analysis = boxing_analyzer.analyze_technique(user_poses, selected_technique)

    #---- Compare videos, frame-by-frame ----#
    comparison = boxing_analyzer.compare_videos(ref_poses, user_poses, ref_frames, user_frames)

    #---- Visualization section ----#
    st.header(f"Comparison Results for '{selected_technique.title()}'")
    col1, col2 = st.columns(2)

    # Show thumbnails of input videos
    user_thumb = video_processor.create_thumbnail(user_video_path, timestamp=1)
    ref_thumb  = video_processor.create_thumbnail(ref_video_path, timestamp=1)
    col1.image(user_thumb, caption="User Video Sample")
    col2.image(ref_thumb, caption="Reference Video Sample")

    # Show accuracy scores and summary
    st.subheader("Technique Accuracy Scores")
    scores_expander = st.expander("Click to Show/Hide Accuracy Details", expanded=True)
    if "joint_accuracies" in analysis and analysis["joint_accuracies"]:
        accuracy_chart = create_accuracy_visualization(analysis["joint_accuracies"])
        scores_expander.write(f"**Overall Accuracy:** {analysis['overall_accuracy']:.1f}%")
        scores_expander.bar_chart(accuracy_chart["accuracies"])
        scores_expander.write("Joint-wise accuracy:")
        for joint, acc in analysis["joint_accuracies"].items():
            scores_expander.write(f"- {joint}: {acc:.1f}%")

    # Show feedback summary
    st.subheader("Detailed Feedback and Corrections")
    feedback_msg = format_feedback_message(analysis.get("feedback", {}))
    st.markdown(feedback_msg)

    # Display difference analysis table
    if "differences" in comparison and comparison["differences"]:
        st.subheader("Pose Differences (Averages across video)")
        diff_table = []
        for joint, info in comparison["differences"].items():
            diff_table.append([joint, f"{info['difference']:.1f}°", info['severity'], info['description']])
        st.table(diff_table)

    # (Optional) Show sample error frames
    st.subheader("Sample Frames with High Differences")
    for i, frame_idx in enumerate([0, len(user_frames)//2, len(user_frames)-1]):
        user_pose_img = pose_estimator.draw_pose(user_frames[frame_idx], user_poses[frame_idx])
        ref_pose_img = pose_estimator.draw_pose(ref_frames[frame_idx], ref_poses[frame_idx])
        st.image([ref_pose_img, user_pose_img], caption=[f"Reference Frame {frame_idx}", f"User Frame {frame_idx}"], width=300)

    st.success("Analysis complete! Review scores and suggested corrections above to improve your boxing skill accuracy.")
else:
    st.write("Please upload both a user video and a reference athlete video to begin analysis.")
