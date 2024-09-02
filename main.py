# -*- coding: utf-8 -*-

### Google Driveと繋げる
"""

from google.colab import drive
drive.mount('/content/drive')

"""### Media-pipeのダウンロード"""

#!pip install mediapipe

"""### 動画ファイルの読み込みする関数"""

def load_video(input_filename):
    video_data = cv2.VideoCapture(input_filename)
    video_width = int(video_data.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = video_data.get(cv2.CAP_PROP_FPS)
    print('FPS:', video_fps)
    print('Dimensions:', video_width, video_height)
    video_data_array = []

    print("VideoFrame:", int(video_data.get(cv2.CAP_PROP_FRAME_COUNT)))

    while video_data.isOpened():
        success, image = video_data.read()
        if success:
            video_data_array.append(image)
        else:
            break
    video_data.release()
    print('Frames Read:', len(video_data_array))

    return video_data_array, video_width, video_height, video_fps

"""### media-pipeの処理をする関数(mp4&csv形式で出力)"""

def process_video_landmarks(video_data_array, video_width, video_height, video_fps, output_filename_mp4, output_filename_csv):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    # MP4出力の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename_mp4, fourcc, video_fps, (video_width, video_height))

    # CSV出力の設定
    with open(output_filename_csv, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # CSVヘッダーの書き込み
        csv_header = ['frame']
        for i in range(468):  # 顔のランドマーク数
            csv_header.extend([f'face_x_{i}', f'face_y_{i}', f'face_z_{i}'])
        for hand in ['left', 'right']:
            for i in range(21):  # 各手のランドマーク数
                csv_header.extend([f'{hand}_hand_x_{i}', f'{hand}_hand_y_{i}', f'{hand}_hand_z_{i}'])
        csv_writer.writerow(csv_header)

        # Holistic modelを使用して顔と手の検出を行う
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for loop_counter, image_data in enumerate(video_data_array):
                # 画像解析
                image_data_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_data_rgb)

                # MP4用の画像処理
                annotated_image = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                out.write(annotated_image)

                # CSVデータの準備
                csv_row = [loop_counter]
                for landmarks in [results.face_landmarks, results.left_hand_landmarks, results.right_hand_landmarks]:
                    if landmarks:
                        csv_row.extend([landmark.x for landmark in landmarks.landmark])
                        csv_row.extend([landmark.y for landmark in landmarks.landmark])
                        csv_row.extend([landmark.z for landmark in landmarks.landmark])
                    else:
                        csv_row.extend([0] * (len(landmarks.landmark) * 3 if landmarks else 468 * 3 if 'face' in str(landmarks) else 21 * 3))

                csv_writer.writerow(csv_row)

                if loop_counter % 100 == 0:
                    print(f"Processed frame {loop_counter}")

    out.release()
    print("Video processing completed. Output saved as", output_filename_mp4)
    print("CSV data saved as", output_filename_csv)

"""### main関数"""

import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import csv

# メイン処理
input_filename = '読み込みをしたい動画pathをここに書く'
output_filename_mp4 = 'output_face_hands_only.mp4'
output_filename_csv = 'landmarks_data.csv'

# ビデオを読み込む
video_data_array, video_width, video_height, video_fps = load_video(input_filename)

# ランドマーク処理を実行し、MP4とCSVを出力
process_video_landmarks(video_data_array, video_width, video_height, video_fps, output_filename_mp4, output_filename_csv)

