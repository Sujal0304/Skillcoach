import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import tempfile
import os

class VideoProcessor:
    def __init__(self):
        """Initialize video processor with optimized settings."""
        self.max_frames = 150  # Limit frames for faster processing
        self.min_frame_width = 480  # Minimum width for pose detection
        
    def extract_frames(self, video_path: str, target_fps: int = 10) -> List[np.ndarray]:
        """
        Extract frames from video at specified frame rate for optimized processing.
        
        Args:
            video_path: Path to the video file
            target_fps: Target frame rate for extraction (default: 10 FPS)
            
        Returns:
            List of extracted frames as numpy arrays
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if original_fps <= 0:
            original_fps = 30  # Default assumption
        
        # Calculate frame skip interval
        skip_frames = max(1, int(original_fps / target_fps))
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames to achieve target FPS
                if frame_count % skip_frames == 0:
                    # Optimize frame for pose detection
                    processed_frame = self._optimize_frame(frame)
                    if processed_frame is not None:
                        frames.append(processed_frame)
                        extracted_count += 1
                    
                    # Limit total frames for processing speed
                    if extracted_count >= self.max_frames:
                        break
                
                frame_count += 1
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
        finally:
            cap.release()
        
        if not frames:
            raise ValueError("No frames could be extracted from the video")
        
        print(f"Extracted {len(frames)} frames from {frame_count} total frames")
        return frames
    
    def _optimize_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Optimize frame for pose detection - resize and enhance if needed.
        
        Args:
            frame: Input frame
            
        Returns:
            Optimized frame or None if frame is invalid
        """
        if frame is None or frame.size == 0:
            return None
        
        height, width = frame.shape[:2]
        
        # Resize frame if too large (for speed) or too small (for accuracy)
        if width > 1280 or width < self.min_frame_width:
            if width < self.min_frame_width:
                # Upscale small frames
                scale_factor = self.min_frame_width / width
            else:
                # Downscale large frames
                scale_factor = 1280 / width
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Enhance contrast for better pose detection
        frame = self._enhance_contrast(frame)
        
        return frame
    
    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance frame contrast for better pose detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information for processing optimization.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with video properties
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'error': 'Unable to open video'}
        
        try:
            info = {
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': 0
            }
            
            # Calculate duration
            if info['fps'] > 0:
                info['duration'] = info['frame_count'] / info['fps']
            
            return info
            
        except Exception as e:
            return {'error': f'Error reading video info: {e}'}
        finally:
            cap.release()
    
    def create_thumbnail(self, video_path: str, timestamp: float = 1.0) -> Optional[np.ndarray]:
        """
        Create a thumbnail from the video at specified timestamp.
        
        Args:
            video_path: Path to the video file
            timestamp: Timestamp in seconds for thumbnail extraction
            
        Returns:
            Thumbnail frame or None if failed
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret:
                # Resize thumbnail to reasonable size
                height, width = frame.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = 640
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                return frame
            
        except Exception as e:
            print(f"Error creating thumbnail: {e}")
        finally:
            cap.release()
        
        return None
    
    def compress_video_for_processing(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Compress video for faster processing while maintaining pose detection quality.
        
        Args:
            video_path: Path to input video
            output_path: Path for compressed output (optional)
            
        Returns:
            Path to compressed video
        """
        if output_path is None:
            # Create temporary file
            fd, output_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")
        
        # Get original video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Set compression parameters
        target_fps = min(15, original_fps)  # Limit to 15 FPS max
        
        # Resize if too large
        if width > 1280:
            scale = 1280 / width
            width = 1280
            height = int(height * scale)
        
        # Set up video writer with compression
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        frame_skip = max(1, int(original_fps / target_fps))
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Resize frame if needed
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    
                    out.write(frame)
                
                frame_count += 1
                
        except Exception as e:
            print(f"Error during compression: {e}")
        finally:
            cap.release()
            out.release()
        
        return output_path
