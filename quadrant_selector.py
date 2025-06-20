import os
import sys
import pandas as pd
import numpy as np
import cv2
import shlex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressBar,
                             QComboBox)
from collections import Counter
import platform
from PyQt5.QtCore import pyqtSignal, QObject
import threading

def determine_quadrant(x, y, width, height):
    """Determine the gaze quadrant based on (x, y) coordinates and screen dimensions."""
    center_x = width / 2
    center_y = height / 2
    axis_x = width / 4
    axis_y = height / 4

    # Check if the point is within the ellipse
    if ((x - center_x) ** 2) / (axis_x ** 2) + ((y - center_y) ** 2) / (axis_y ** 2) <= 1:
        return 'center'

    slope = height / width
    if y < slope * x:  # Above top-left to bottom-right
        if y < height - slope * x:  # Above top-right to bottom-left
            return 'up'
        else:
            return 'right'
    else:  # Below top-left to bottom-right
        if y < height - slope * x:  # Below top-right to bottom-left
            return 'left'
        else:
            return 'down'


def get_openface_binary():
    if platform.system() == "Windows":
        # Adjust this path as needed for your Windows setup
        return os.path.join("openFace", "OpenFace", "build", "bin", "FeatureExtraction.exe")
    else:
        return os.path.join("openFace", "OpenFace", "build", "bin", "FeatureExtraction")


def process_videos(input_dir, output_dir, progress_callback, stability_threshold=0.5):
    """Process videos for gaze quadrant analysis and overlay.
    
    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save output files
        progress_callback: Function to call with progress updates
        stability_threshold: Time in seconds that gaze must remain in the same quadrant
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all videos from the input directory
    videos = [video for video in os.listdir(input_dir) if video.endswith(('.mp4', '.avi', '.mov'))]

    total_videos = len(videos)
    processed_videos = 0

    # Run openface on these videos
    openface_binary = get_openface_binary()
    for video in videos:
        video_path = os.path.join(input_dir, video)
        output_path = output_dir
        if platform.system() == "Windows":
            # No shlex.quote on Windows
            cmd = f'"{openface_binary}" -f "{video_path}" -out_dir "{output_path}" -gaze -tracked'
        else:
            cmd = f'{shlex.quote(openface_binary)} -f {shlex.quote(video_path)} -out_dir {shlex.quote(output_path)} -gaze -tracked'
        result = os.system(cmd)
        if result != 0:
            print(f"OpenFace failed for {video}. Command: {cmd}")
        processed_videos += 1
        progress_callback(int((processed_videos / total_videos) * 100))

    # Find matching csv from openface output
    openface_files = os.listdir(output_dir)
    csvs = [file for file in openface_files if '.csv' in file]
    videos_of = [file for file in openface_files if '.avi' in file]

    # For each csv, perform gaze quadrant analysis
    for csv in csvs:
        # Read csv
        df = pd.read_csv(os.path.join(output_dir, csv))

        # Create two eye vectors for data gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y
        eye_vector_0 = df[['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_angle_x', 'gaze_angle_y']].apply(lambda row: np.array(row), axis=1)
        eye_vector_1 = df[['gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']].apply(lambda row: np.array(row), axis=1)

        # Condense the eye vectors into a single vector
        df['condensed_eye_vector'] = [(v0[:3] + v1[:3]) / 2 for v0, v1 in zip(eye_vector_0, eye_vector_1)]
        df['condensed_gaze_angle'] = [(v0[3:] + v1[3:]) / 2 for v0, v1 in zip(eye_vector_0, eye_vector_1)]

        # Interpolate the gaze point based on the direction vector
        def interpolate_gaze_point(row):
            x, y, z = row['condensed_eye_vector']
            angle_x, angle_y = row['condensed_gaze_angle']
            # Assuming a fixed distance to the screen (e.g., 1 unit)
            distance = 1
            gaze_x = x + distance * np.tan(angle_x)
            gaze_y = y + distance * np.tan(angle_y)
            return gaze_x, gaze_y

        df['gaze_x'], df['gaze_y'] = zip(*df.apply(interpolate_gaze_point, axis=1))

        df['gaze_quadrant'] = df.apply(lambda row: determine_quadrant(
            row['gaze_x'], row['gaze_y'], 1, 1), axis=1)  # Assuming normalized coordinates (0-1)

        # Output the gaze quadrant to a CSV file
        df.to_csv(os.path.join(output_dir, 'gaze_' + csv), index=False)

    # Overlay on the output video from OpenFace
    for video in videos_of:
        video_file = os.path.join(output_dir, video)
        # Find the corresponding CSV file
        csv_file = os.path.join(output_dir, 'gaze_' + video.replace('.avi', '.csv'))

        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} does not exist. Skipping video {video}.")
            continue

        # Read the CSV file
        gaze_data = pd.read_csv(csv_file)

        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video file {video_file}.")
            continue

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create VideoWriter object to save the output video
        output_file = os.path.join(output_dir, 'overlay_' + video.replace('.avi', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Failed to create VideoWriter object for {output_file}.")
            cap.release()
            continue

        # Calculate how many frames represent 2 seconds
        frames_in_2_seconds = int(fps * 2)
        
        # Calculate how many frames represent the selected stability threshold
        frames_for_stability = int(fps * stability_threshold)
        
        # Initialize a list to store the quadrants from the last 2 seconds
        recent_quadrants = []
        
        # Variables to track stable quadrants
        current_stable_quadrant = None
        consecutive_same_quadrant_count = 0
        
        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the corresponding gaze quadrant info
            if frame_index < len(gaze_data):
                gaze_quadrant = gaze_data.iloc[frame_index]['gaze_quadrant']
                gaze_x = gaze_data.iloc[frame_index]['gaze_x']
                gaze_y = gaze_data.iloc[frame_index]['gaze_y']
                
                # Track consecutive frames with the same quadrant
                if gaze_quadrant == current_stable_quadrant:
                    consecutive_same_quadrant_count += 1
                else:
                    current_stable_quadrant = gaze_quadrant
                    consecutive_same_quadrant_count = 1
                
                # Only consider the quadrant if it's been stable for the selected time threshold
                stable_quadrant = None
                if consecutive_same_quadrant_count >= frames_for_stability:
                    stable_quadrant = current_stable_quadrant
                    
                    # Only update the sliding window if the quadrant is stable
                    recent_quadrants.append(stable_quadrant)
                    if len(recent_quadrants) > frames_in_2_seconds:
                        recent_quadrants.pop(0)  # Remove oldest entry
                
                # Calculate the most frequent quadrant in the last 2 seconds
                if recent_quadrants:
                    # Filter to only include left and right quadrants
                    left_right_quadrants = [q for q in recent_quadrants if q in ['left', 'right']]
                    
                    if left_right_quadrants:
                        counter = Counter(left_right_quadrants)
                        most_common_quadrant = counter.most_common(1)[0][0]
                        quadrant_count = counter.most_common(1)[0][1]
                        quadrant_percentage = (quadrant_count / len(left_right_quadrants)) * 100
                    else:
                        most_common_quadrant = "none"
                        quadrant_percentage = 0
                else:
                    most_common_quadrant = "unknown"
                    quadrant_percentage = 0

                # Draw translucent rectangle over the quadrant
                overlay = frame.copy()
                alpha = 0.6  # Increased transparency factor for brighter highlights

                # Draw the lines from top left to bottom right and top right to bottom left
                cv2.line(overlay, (0, 0), (frame_width, frame_height), (0, 0, 255), 4)
                cv2.line(overlay, (frame_width, 0), (0, frame_height), (0, 0, 255), 4)

                # Draw the center ellipse
                center_x = frame_width // 2
                center_y = frame_height // 2
                axis_x = frame_width // 4
                axis_y = frame_height // 4
                cv2.ellipse(overlay, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, (0, 0, 255), 4)

                # Draw the words "UP", "DOWN", "LEFT", "RIGHT"
                text_color = (255, 255, 255)  # White color for default
                highlight_color = (0, 255, 0)  # Green color for highlight

                cv2.putText(overlay, 'UP', (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4, cv2.LINE_AA)
                cv2.putText(overlay, 'DOWN', (frame_width // 2 - 100, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4, cv2.LINE_AA)
                cv2.putText(overlay, 'LEFT', (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4, cv2.LINE_AA)
                cv2.putText(overlay, 'RIGHT', (frame_width - 200, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, text_color, 4, cv2.LINE_AA)

                # Create a mask for the center ellipse
                center_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                cv2.ellipse(center_mask, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 255, -1)

                # Only highlight and emphasize quadrant text if it's been stable for the required time
                if consecutive_same_quadrant_count >= frames_for_stability:
                    if gaze_quadrant == 'center':
                        # For center, create an elliptical highlight
                        highlight_mask = center_mask.copy()
                        highlight_overlay = np.zeros_like(overlay)
                        highlight_overlay[center_mask > 0] = (0, 255, 255)  # Cyan color (BGR format)
                        cv2.addWeighted(highlight_overlay, alpha, overlay, 1, 0, overlay)

                    else:
                        # For directional quadrants, create polygon that excludes the center ellipse
                        highlight_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                        if gaze_quadrant == 'up':
                            points = np.array([[0, 0], [frame_width, 0], [frame_width // 2, frame_height // 2]])
                            cv2.putText(overlay, 'UP', (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                        elif gaze_quadrant == 'down':
                            points = np.array([[0, frame_height], [frame_width, frame_height], [frame_width // 2, frame_height // 2]])
                            cv2.putText(overlay, 'DOWN', (frame_width // 2 - 100, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                        elif gaze_quadrant == 'left':
                            points = np.array([[0, 0], [0, frame_height], [frame_width // 2, frame_height // 2]])
                            cv2.putText(overlay, 'LEFT', (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                        elif gaze_quadrant == 'right':
                            points = np.array([[frame_width, 0], [frame_width, frame_height], [frame_width // 2, frame_height // 2]])
                            cv2.putText(overlay, 'RIGHT', (frame_width - 200, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)

                        # Fill the polygon on the mask
                        cv2.fillPoly(highlight_mask, [points], 255)

                        # Subtract the center ellipse from the highlight mask
                        highlight_mask = cv2.subtract(highlight_mask, center_mask)

                        # Apply the highlight color using the mask
                        highlight_overlay = np.zeros_like(overlay)
                        highlight_overlay[highlight_mask > 0] = (0, 255, 255)  # Cyan color (BGR format)
                        cv2.addWeighted(highlight_overlay, alpha, overlay, 1, 0, overlay)

                # Blend the overlay with the frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # overlay gaze point
                cv2.circle(frame, (int(gaze_x * frame_width), int(gaze_y * frame_height)), 10, (0, 0, 0), -1)

                # Add text showing the most frequent quadrant over the last 2 seconds
                stats_text = f"Most frequent (2s): {most_common_quadrant.upper()} ({quadrant_percentage:.1f}%)"
                cv2.rectangle(frame, (10, frame_height - 80), (500, frame_height - 20), (0, 0, 0), -1)
                cv2.putText(
                    frame, stats_text, (20, frame_height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )

            # Write the frame to the output video
            out.write(frame)
            frame_index += 1

        # Release the video objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        processed_videos += 1
        progress_callback(int((processed_videos / total_videos) * 100))

        print(f"Processed video {video} and saved to {output_file}.")


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, input_dir, output_dir, stability_threshold):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.stability_threshold = stability_threshold

    def run(self):
        process_videos(self.input_dir, self.output_dir, self.progress.emit, self.stability_threshold)
        self.finished.emit()


class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Processing")
        self.setGeometry(100, 100, 600, 200)

        # Create layout
        layout = QVBoxLayout()

        # Input directory
        self.input_label = QLabel("Video(s) Directory:")
        self.input_line_edit = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_dir)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(self.input_button)

        layout.addWidget(self.input_label)
        layout.addLayout(input_layout)

        # Output directory
        self.output_label = QLabel("Output Directory:")
        self.output_line_edit = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_dir)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(self.output_button)

        layout.addWidget(self.output_label)
        layout.addLayout(output_layout)

        # Add stability threshold selection
        self.stability_label = QLabel("Stability Threshold (seconds):")
        self.stability_combo = QComboBox()
        for time in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            self.stability_combo.addItem(str(time))
        # Set default to 0.5 seconds
        self.stability_combo.setCurrentText("0.5")
        
        stability_layout = QHBoxLayout()
        stability_layout.addWidget(self.stability_label)
        stability_layout.addWidget(self.stability_combo)
        stability_layout.addStretch()
        
        layout.addLayout(stability_layout)

        # Start processing button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        layout.addWidget(self.process_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Set central widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.worker = None
        self.worker_thread = None

    def select_input_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Video(s) Directory")
        if directory:
            self.input_line_edit.setText(directory)

    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_line_edit.setText(directory)

    def start_processing(self):
        input_dir = self.input_line_edit.text()
        output_dir = self.output_line_edit.text()
        stability_threshold = float(self.stability_combo.currentText())

        if not input_dir:
            QMessageBox.critical(self, "Error", "Please select an input directory.")
            return

        self.progress_bar.setValue(0)
        self.process_button.setEnabled(False)
        self.worker = Worker(input_dir, output_dir, stability_threshold)
        self.worker_thread = threading.Thread(target=self.worker.run, daemon=True)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.processing_finished)
        self.worker_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self):
        self.process_button.setEnabled(True)
        QMessageBox.information(self, "Success", "Video processing completed.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())