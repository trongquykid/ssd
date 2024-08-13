import onnxruntime as ort
import numpy as np
import cv2
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from ultralytics import YOLO
from detect_character import recognition_charv2, arrange_correspond
import time
from tracker import Tracker
from collections import Counter
import os
import random

net_type = 'mb1-ssd-lite'
model_path = 'models/plate/az_plate/plate_v1_ssdmobilenetv2.onnx'
model_ocr= YOLO('models/ocr/yolo_ocr_detection_v2.pt')
label_path = 'models/plate/az_plate/labels.txt'
label_path_ocr = 'models/ocr/labels.txt'
image_path = 'ImageQuy_1.jpg'

# Tải các tên lớp
class_names = [name.strip() for name in open(label_path).readlines()]
class_name_ocr = [name.strip() for name in open(label_path_ocr).readlines()]

session = ort.InferenceSession(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(session, candidate_size=200)

def add_character(list_label, list_box, list_character):

    for i in range(len(list_box)):
        label_ocr = int(list_character[i])
        list_label.append(class_name_ocr[label_ocr])

def add_probs(list_label, list_box, list_probs):

    for i in range(len(list_box)):
        char_probs = round(float(list_probs[i]), 2)
        list_label.append(char_probs)

def rotate_license_plate(image):

    if image is None or image.size == 0:
        print("Empty image received")
        return image
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_image = cv2.Canny(gray_image, 100, 200, apertureSize=3, L2gradient=True)
    
    # Phát hiện các đoạn thẳng dài và thẳng bằng Hough Line Transform
    lines = cv2.HoughLines(edges_image, 1, np.pi / 180, threshold=100)

    if lines is not None:
        # Lọc và chọn lựa đoạn thẳng phù hợp (đoạn thẳng có độ dài > threshold_length)
        threshold_length = 80
        filtered_lines = []
        for line in lines:
            rho, theta = line[0]
            if np.pi / 4 < theta < 3 * np.pi / 4:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * a)
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if line_length > threshold_length:
                    filtered_lines.append(((x1, y1), (x2, y2)))

                # Vẽ các đường thẳng
                # cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # print("filtered_lines:", filtered_lines)
        if len(filtered_lines) > 0:
            # Lựa chọn đoạn thẳng dài nhất
            longest_line = max(filtered_lines, key=lambda x: np.linalg.norm(np.array(x[0]) - np.array(x[1])))
            x1, y1 = longest_line[0]
            x2, y2 = longest_line[1]

            # Tính góc xoay của đường thẳng
            rotation_angle = (np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            # Xoay lại ảnh để biển số xe nằm ngang
            height, width = image.shape[:2]
            print("height, width:", height, width)
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

            return rotated_image
    else:
        print("NO LINES!")
        return image
    
def get_best_ocr(data, track_ids):
    counter_dict = Counter((item['track_id'], item['ocr_txt']) for item in data)

    most_common_recognized_text = {}
    rec_conf = ""
    ocr_res = ""
    for item in data:
        track_id = item['track_id']
        recognized_text = item['ocr_txt']
        confidence = item['ocr_conf']
        count = counter_dict[(track_id, recognized_text)]

        current_count, current_confidence, current_text = most_common_recognized_text.get(track_id, (0, 0, ''))

        if count > current_count or (count == current_count and confidence > current_confidence):
            most_common_recognized_text[track_id] = (count, confidence, recognized_text)

    if track_ids in most_common_recognized_text:
        rec_conf, ocr_res = most_common_recognized_text[track_ids][1], most_common_recognized_text[track_ids][2]

    return rec_conf, ocr_res

def save_info_file(name_folder, name_file, data):
    # Check if the directory does not exist then create a new one
    if not os.path.exists(name_folder):
        os.makedirs(name_folder)

    # Create the full path to the file
    file_path = os.path.join(name_folder, f"{name_file}.txt")

    with open(file_path, 'a') as file:
        
        # Add data to the file
        for entry in data:
            file.write(f"{entry['Track_id']}\t{entry['Recognized_text']}\t{entry['Confidence']}\t{entry['Aspect_ratio']}\n")
            # file.write(f"{entry['Track_id']}\t{entry['Recognized_text']}\t{entry['Confidence']}\n")

    print("Data has been added to the file.")

def process_frame_tracker(frame, tracker, preds):
    boxes, labels, probs = predictor.predict(frame, 10, 0.4)
    result_tracker = []

    for i in range(boxes.size(0)):
        box = boxes[i]
        score = probs[i]
        x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        result_tracker.append([x_min, y_min, x_max, y_max, score])

    tracker.update(frame, result_tracker)

    if boxes is not None and len(boxes) > 0:
        for track in tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.bbox
            width = x2 - x1
            height = y2 - y1

            # Check for valid box dimensions
            if width <= 0 or height <= 0:
                continue

            aspect_ratio = width / height
            label_list = {'recognition_text': [], 'confidence': []}

            if 0 <= aspect_ratio <= 2.0:
                plate = frame[int(y1):int(y2), int(x1):int(x2)]

                plate = rotate_license_plate(plate)

                if plate is None:
                    continue

                if plate.size == 0:  # Check if the plate is empty
                    continue

                results = model_ocr(plate)

                for result in results:
                    index_char = result.boxes.cls
                    conf_char = result.boxes.conf
                    box_char = result.boxes.xyxy

                    (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = recognition_charv2(index_char, conf_char, box_char)

                    add_character(label_list['recognition_text'], up_list, label_up_list)
                    add_probs(label_list['confidence'], up_list, probs_up_list)

                    add_character(label_list['recognition_text'], low_list, label_low_list)
                    add_probs(label_list['confidence'], low_list, probs_low_list)
            else:
                plate = frame[int(y1):int(y2), int(x1):int(x2)]

                plate = rotate_license_plate(plate)

                if plate is None:
                    continue

                if plate.size == 0:  # Check if the plate is empty
                    continue

                results = model_ocr(plate)

                for result in results:
                    index_char = result.boxes.cls
                    conf_char = result.boxes.conf
                    box_char = result.boxes.xyxy

                    up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                    add_character(label_list['recognition_text'], up_list, label_up_list)
                    add_probs(label_list['confidence'], up_list, probs_up_list)

            label_text_ocr = ''.join(label_list['recognition_text'])
            if len(label_list['confidence']) != 0:
                confidence_text = round(sum(label_list['confidence']) / len(label_list['confidence']), 2)
            else:
                confidence_text = ""
            output_frame = {"track_id": track_id, "ocr_txt": label_text_ocr, "ocr_conf": confidence_text}
            preds.append(output_frame)

            if track_id in list(set(ele['track_id'] for ele in preds)):
                rec_conf, ocr_resc = get_best_ocr(preds, track_id)
                save_info_file(f"./save_file_txt", 'log', [{'Track_id': track_id, 'Recognized_text': ocr_resc, 'Confidence': rec_conf, 'Aspect_ratio': aspect_ratio}])
            txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame    

def process_frame(frame):
    boxes, labels, probs = predictor.predict(frame, 10, 0.6)

    # Xử lý sau và vẽ các hộp lên ảnh gốc
    for i in range(boxes.size(0)):
        box = boxes[i]
        label = int(labels[i])
        score = probs[i]
        width = box[2] - box[0]
        height = box[3] - box[1]

        # Check for valid box dimensions
        if width <= 0 or height <= 0:
            continue

        aspect_ratio = width / height

        label_list = []

        if 0 <= aspect_ratio <= 2.0:
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
            if plate.size == 0:  # Check if the plate is empty
                continue

            results = model_ocr(plate)

            for result in results:
                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
                
                (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = recognition_charv2(index_char, conf_char, box_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])
                    label_list.append(class_name_ocr[label_ocr]) 

                for j in range(len(low_list)):
                    label_ocr = int(label_low_list[j])
                    label_list.append(class_name_ocr[label_ocr])
        else:
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
            if plate.size == 0:  # Check if the plate is empty
                continue
            
            results = model_ocr(plate)
            
            for result in results:
                label_list = []

                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
        
                up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                for j in range(len(up_list)):
                    label_ocr = int(label_up_list[j])
                    label_list.append(class_name_ocr[label_ocr])

        label_text_ocr = ''.join(label_list)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        cv2.putText(frame, label_text_ocr, (int(box[0]) + 20, int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    return frame

def process_camera():
    cap = cv2.VideoCapture(0)  # Open the default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    tracker = Tracker()
    preds = []
    frame_number = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image")
            break

        prev_time = time.time()
        img_tmp = frame
        processed_frame = process_frame_tracker(frame, tracker, preds)

        tot_time = time.time() - prev_time
        fps = round(1 / tot_time, 2)

        cv2.putText(img_tmp, 'frame: %d fps: %s' % (frame_number, fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1
        print(f'Processing frame {frame_number}', end='\r')
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nProcessing complete.")

# Start processing the camera feed
process_camera()
