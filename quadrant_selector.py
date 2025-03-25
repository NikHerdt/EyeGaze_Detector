import os
import sys
import pandas as pd
import numpy as np
import cv2
import shlex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressBar)

def get_normal_vector(row):
    """Compute the normal vector from head pose angles."""
    rx, ry, rz = row['pose_Rx'], row['pose_Ry'], row['pose_Rz']
    
    # Convert Euler angles to a direction vector
    normal_vector = np.array([
        np.cos(ry) * np.cos(rx),  # X component
        np.sin(rx),               # Y component
        np.sin(ry) * np.cos(rx)   # Z component
    ])
    
    return normal_vector

def get_combined_gaze_vector(row):
    """Compute the combined gaze vector from left and right eye gaze vectors."""
    # Check if gaze data is available
    if 'gaze_0_x' in row and 'gaze_1_x' in row:
        # Average the left and right eye gaze vectors
        gaze_vector = np.array([
            (row['gaze_0_x'] + row['gaze_1_x']) / 2,  # X component
            (row['gaze_0_y'] + row['gaze_1_y']) / 2,  # Y component
            (row['gaze_0_z'] + row['gaze_1_z']) / 2   # Z component
        ])
        return gaze_vector
    else:
        # Return a default forward-facing vector if no gaze data
        return np.array([0, 0, 1])

def determine_quadrant_from_normal(normal_vector, eyegaze_vector=None):
    """Determine the gaze quadrant based on the relation between gaze and normal vector."""
    # If no eyegaze vector provided, use only the normal vector (backward compatibility)
    if eyegaze_vector is None:
        x, y, z = normal_vector
        
        if y > abs(x) and y > abs(z):
            return 'up'
        elif -y > abs(x) and -y > abs(z):
            return 'down'
        elif x > abs(y) and x > abs(z):
            return 'right'
        elif -x > abs(y) and -x > abs(z):
            return 'left'
        return 'center'
    
    # Calculate the relative direction between gaze and normal
    # This gets the gaze vector relative to head orientation
    relative_x = eyegaze_vector[0] - normal_vector[0]
    relative_y = eyegaze_vector[1] - normal_vector[1]
    
    # Determine horizontal component
    if abs(relative_x) < 0.2:  # Threshold for center
        h_direction = 'center'
    elif relative_x > 0:
        h_direction = 'right'
    else:
        h_direction = 'left'
        
    # Determine vertical component
    if abs(relative_y) < 0.2:  # Threshold for center
        v_direction = 'center'
    elif relative_y > 0:
        v_direction = 'up'
    else:
        v_direction = 'down'
    
    # Combine directions
    if h_direction == 'center' and v_direction == 'center':
        return 'center'
    elif h_direction == 'center':
        return v_direction
    elif v_direction == 'center':
        return h_direction
    else:
        return f"{v_direction}_{h_direction}"

def process_videos(input_dir, output_dir, progress_callback):
    """Process videos for gaze quadrant analysis using head pose normal vector."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos = os.listdir(input_dir)
    videos_r = [video for video in videos if 'R' in video]

    total_videos = len(videos_r)
    processed_videos = 0

    # Run OpenFace on the videos
    for video in videos_r:
        video_path = shlex.quote(os.path.join(input_dir, video))
        output_path = shlex.quote(output_dir)
        os.system(f'OpenFace/build/bin/FeatureExtraction -f {video_path} -out_dir {output_path}')
        processed_videos += 1
        progress_callback(int((processed_videos / total_videos) * 100))

    openface_files = os.listdir(output_dir)
    csvs = [file for file in openface_files if file.endswith('.csv')]
    videos_of = [file for file in openface_files if file.endswith('.avi')]

    for csv in csvs:
        df = pd.read_csv(os.path.join(output_dir, csv))

        if 'pose_Rx' not in df.columns or 'pose_Ry' not in df.columns or 'pose_Rz' not in df.columns:
            print(f"Missing head pose data in {csv}. Skipping.")
            continue
        
        # Check if gaze data columns exist
        has_gaze_data = all(col in df.columns for col in ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 
                                                         'gaze_1_x', 'gaze_1_y', 'gaze_1_z'])
        
        df['head_normal_vector'] = df.apply(get_normal_vector, axis=1)
        
        if has_gaze_data:
            df['combined_gaze_vector'] = df.apply(get_combined_gaze_vector, axis=1)
            # Use both vectors to determine quadrant
            df['gaze_quadrant'] = df.apply(lambda row: determine_quadrant_from_normal(
                row['head_normal_vector'], row['combined_gaze_vector']), axis=1)
        else:
            print(f"Warning: No gaze data in {csv}. Using only head pose.")
            df['gaze_quadrant'] = df['head_normal_vector'].apply(determine_quadrant_from_normal)

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

        frame_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Get the corresponding gaze quadrant info
            if frame_index < len(gaze_data):
                gaze_quadrant = gaze_data.iloc[frame_index]['gaze_quadrant']

                # Draw translucent rectangle over the quadrant
                overlay = frame.copy()
                alpha = 0.3  # Transparency factor

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

                # Update to handle combined quadrants
                if gaze_quadrant == 'up':
                    points = np.array([[0, 0], [frame_width, 0], [frame_width // 2, frame_height // 2]])  # Upper triangle
                    cv2.putText(overlay, 'UP', (frame_width // 2 - 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'down':
                    points = np.array([[0, frame_height], [frame_width, frame_height], [frame_width // 2, frame_height // 2]])  # Lower triangle
                    cv2.putText(overlay, 'DOWN', (frame_width // 2 - 100, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'left':
                    points = np.array([[0, 0], [0, frame_height], [frame_width // 2, frame_height // 2]])  # Left triangle
                    cv2.putText(overlay, 'LEFT', (50, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'right':
                    points = np.array([[frame_width, 0], [frame_width, frame_height], [frame_width // 2, frame_height // 2]])  # Right triangle
                    cv2.putText(overlay, 'RIGHT', (frame_width - 200, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'center':
                    # set points to an ellipse in the center
                    points = np.array([[center_x - axis_x, center_y - axis_y], [center_x + axis_x, center_y - axis_y],
                                       [center_x + axis_x, center_y + axis_y], [center_x - axis_x, center_y + axis_y]])
                elif gaze_quadrant == 'up_left':
                    # Top-left quadrant
                    points = np.array([[0, 0], [frame_width//2, 0], [frame_width//2, frame_height//2], [0, frame_height//2]])
                    cv2.putText(overlay, 'UP_LEFT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'up_right':
                    # Top-right quadrant
                    points = np.array([[frame_width//2, 0], [frame_width, 0], [frame_width, frame_height//2], [frame_width//2, frame_height//2]])
                    cv2.putText(overlay, 'UP_RIGHT', (frame_width-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'down_left':
                    # Bottom-left quadrant
                    points = np.array([[0, frame_height//2], [frame_width//2, frame_height//2], [frame_width//2, frame_height], [0, frame_height]])
                    cv2.putText(overlay, 'DOWN_LEFT', (50, frame_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, highlight_color, 4, cv2.LINE_AA)
                elif gaze_quadrant == 'down_right':
                    # Bottom-right quadrant
                    points = np.array([[frame_width//2, frame_height//2], [frame_width, frame_height//2], [frame_width, frame_height], [frame_width//2, frame_height]])
                    cv2.putText(overlay, 'DOWN_RIGHT', (frame_width-250, frame_height-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, highlight_color, 4, cv2.LINE_AA)

                cv2.fillPoly(overlay, [points], (0, 255, 255))

                # Blend the overlay with the frame
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

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

    progress_callback(100)
    print("Processing complete.")

class VideoProcessor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Processing")
        self.setGeometry(100, 100, 600, 200)

        layout = QVBoxLayout()

        self.input_label = QLabel("Video(s) Directory:")
        self.input_line_edit = QLineEdit()
        self.input_button = QPushButton("Browse")
        self.input_button.clicked.connect(self.select_input_dir)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_line_edit)
        input_layout.addWidget(self.input_button)

        layout.addWidget(self.input_label)
        layout.addLayout(input_layout)

        self.output_label = QLabel("Output Directory:")
        self.output_line_edit = QLineEdit()
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.select_output_dir)

        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_line_edit)
        output_layout.addWidget(self.output_button)

        layout.addWidget(self.output_label)
        layout.addLayout(output_layout)

        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)
        layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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

        if not input_dir:
            QMessageBox.critical(self, "Error", "Please select an input directory.")
            return

        self.progress_bar.setValue(0)
        process_videos(input_dir, output_dir, self.update_progress)
        QMessageBox.information(self, "Success", "Video processing completed.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoProcessor()
    window.show()
    sys.exit(app.exec_())