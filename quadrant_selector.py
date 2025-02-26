import os
import sys
import pandas as pd
import numpy as np
import cv2
import shlex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QLineEdit, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressBar)

def determine_quadrant(x, y, width, height):
    """Determine the gaze quadrant based on (x, y) coordinates and screen dimensions."""
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

# Update the process_videos function to pass actual height and width
def process_videos(input_dir, output_dir, progress_callback):
    """Process videos for gaze quadrant analysis and overlay."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videos = os.listdir(input_dir)
    videos_r = [video for video in videos if 'R' in video]

    total_videos = len(videos_r)
    processed_videos = 0

    for video in videos_r:
        video_path = shlex.quote(os.path.join(input_dir, video))
        output_path = shlex.quote(output_dir)
        os.system(f'OpenFace/build/bin/FeatureExtraction -f {video_path} -out_dir {output_path} -gaze -tracked')
        processed_videos += 1
        progress_callback(int((processed_videos / total_videos) * 100))

    openface_files = os.listdir(output_dir)
    csvs = [file for file in openface_files if '.csv' in file]
    videos_of = [file for file in openface_files if '.avi' in file]

    for csv in csvs:
        df = pd.read_csv(os.path.join(output_dir, csv))
        eye_vector_0 = df[['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_angle_x', 'gaze_angle_y']].apply(lambda row: np.array(row), axis=1)
        eye_vector_1 = df[['gaze_1_x', 'gaze_1_y', 'gaze_1_z', 'gaze_angle_x', 'gaze_angle_y']].apply(lambda row: np.array(row), axis=1)

        df['condensed_eye_vector'] = [(v0[:3] + v1[:3]) / 2 for v0, v1 in zip(eye_vector_0, eye_vector_1)]
        df['condensed_gaze_angle'] = [(v0[3:] + v1[3:]) / 2 for v0, v1 in zip(eye_vector_0, eye_vector_1)]

        def interpolate_gaze_point(row):
            x, y, z = row['condensed_eye_vector']
            angle_x, angle_y = row['condensed_gaze_angle']
            distance = 1
            gaze_x = x + distance * np.tan(angle_x)
            gaze_y = y + distance * np.tan(angle_y)
            return gaze_x, gaze_y

        df['gaze_x'], df['gaze_y'] = zip(*df.apply(interpolate_gaze_point, axis=1))

        video_file = os.path.join(output_dir, csv.replace('.csv', '.avi'))
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video file {video_file}.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        df['gaze_quadrant'] = df.apply(lambda row: determine_quadrant(
            row['gaze_x'], row['gaze_y'], frame_width, frame_height), axis=1)

        df.to_csv(os.path.join(output_dir, 'gaze_' + csv), index=False)

    for video in videos_of:
        video_file = os.path.join(output_dir, video)
        csv_file = os.path.join(output_dir, 'gaze_' + video.replace('.avi', '.csv'))

        if not os.path.exists(csv_file):
            print(f"CSV file {csv_file} does not exist. Skipping video {video}.")
            continue

        gaze_data = pd.read_csv(csv_file)
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Failed to open video file {video_file}.")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        output_file = os.path.join(output_dir, 'overlay_' + video.replace('.avi', '.mp4'))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Failed to create VideoWriter object for {output_file}.")
            cap.release()
            continue

        frame_index = 0
        slope = frame_height / frame_width

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index < len(gaze_data):
                gaze_quadrant = gaze_data.iloc[frame_index]['gaze_quadrant']
                overlay = frame.copy()
                alpha = 0.3

                cv2.line(frame, (0, int(slope * frame_width)), (frame_width, 0), (255, 0, 0), 2)
                cv2.line(frame, (0, frame_height - int(slope * frame_width)), (frame_width, frame_height), (255, 0, 0), 2)

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()
        cv2.destroyAllWindows()

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
