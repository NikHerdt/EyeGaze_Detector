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

def determine_quadrant_from_normal(normal_vector):
    """Determine the gaze quadrant based on the head normal vector."""
    x, y, z = normal_vector  # Extract components
    
    if y > abs(x) and y > abs(z):
        return 'up'
    elif -y > abs(x) and -y > abs(z):
        return 'down'
    elif x > abs(y) and x > abs(z):
        return 'right'
    elif -x > abs(y) and -x > abs(z):
        return 'left'
    return 'center'

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

    for csv in csvs:
        df = pd.read_csv(os.path.join(output_dir, csv))

        if 'pose_Rx' not in df.columns or 'pose_Ry' not in df.columns or 'pose_Rz' not in df.columns:
            print(f"Missing head pose data in {csv}. Skipping.")
            continue

        df['head_normal_vector'] = df.apply(get_normal_vector, axis=1)
        df['gaze_quadrant'] = df['head_normal_vector'].apply(determine_quadrant_from_normal)

        df.to_csv(os.path.join(output_dir, 'gaze_' + csv), index=False)

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
