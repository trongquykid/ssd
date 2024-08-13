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

from x64.darknet_image_test import perform_detection

net_type = 'mb1-ssd-lite'
model_path = 'ssd/models/plate/az_plate/plate_v1_ssdmobilenetv2.onnx'
model_ocr= YOLO('ssd/models/ocr/yolo_ocr_detection_v2.pt')
label_path = 'ssd/models/plate/az_plate/labels.txt'
label_path_ocr = 'ssd/models/ocr/labels.txt'
image_path = 'ssd/AQUA2__checkout_2020-10-28-8-38Mtsln3KIhA.jpg'


# Tải các tên lớp
class_names = [name.strip() for name in open(label_path).readlines()]
class_name_ocr = [name.strip() for name in open(label_path_ocr).readlines()]

# Tải mô hình ONNX
session = ort.InferenceSession(model_path)

predictor = create_mobilenetv2_ssd_lite_predictor(session, candidate_size=200)

# Chuẩn bị ảnh đầu vào
orig_image = cv2.imread(image_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

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

def random_crop(image):
    
    original_width, original_height = image.shape[1], image.shape[0]
    x_center,y_center = original_height//2, original_width//2
    
    x_left = random.randint(0, x_center//2)
    x_right = random.randint(original_width-x_center//2, original_width)
    
    y_top = random.randint(0, y_center//2)
    y_bottom = random.randint(original_height-y_center//2, original_width)
    
    # crop ra vùng ảnh với kích thước ngẫu nhiên
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    # resize ảnh bằng kích thước ảnh ban đầu 
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    return cropped_image

def test_yolo_process(path):
    list_image = os.listdir(path)
    start = time.time()
    for image_name in list_image:
        start_image = time.time()
        image_path = f"{path}/{image_name}"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(img, 10, 0.4)

        # Xử lý sau và vẽ các hộp lên ảnh gốc
        if len(boxes) > 0:
            for i in range(boxes.size(0)):
                box = boxes[i]
                label = int(labels[i])
                score = probs[i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = width / height
                # print(aspect_ratio)
                label_list = {'recognition_text' : [],
                            'confidence' : []}
                start_ocr = time.time()
                if 0 <= aspect_ratio <= 2.0:
                    plate = img[int(box[1]) : int(box[3]),int(box[0]) : int(box[2])]

                    # plate= rotate_license_plate(plate)
                    # gray_plate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    # img = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
                    # img = cv2.equalizeHist(img)
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # img = cv2.GaussianBlur(plate, (3, 3), 0)
                    index_char = []
                    conf_char = []
                    box_char = []
                    if plate.shape[0] > 0 and plate.shape[1] > 0:
                        # results = model_ocr(plate, conf=0.6)
                        _, results = perform_detection(image_path=plate)
                        for label, confidence, bbox in results:
                            x, y, w, h = bbox
                            conf= round(float(confidence),2)
                            index_char.append(label)
                            conf_char.append(conf)
                            box_char.append(bbox)
                            
                        for box, label, conf in zip(box_char, index_char, conf_char):
                            x1, y1, x2, y2 = box
                            conf= round(float(conf),2)
                            label_str = f'{label}:{conf}'
                            cv2.rectangle(plate, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
                            cv2.putText(plate, str(label_str),
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,  # font scale
                            (255, 0, 255),
                            2)  # line type
                        (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list)= recognition_charv2(index_char, conf_char, box_char)

                        add_character(label_list['recognition_text'], up_list, label_up_list)
                        add_probs(label_list['confidence'], up_list, probs_up_list)

                        add_character(label_list['recognition_text'], low_list, label_low_list)
                        add_probs(label_list['confidence'], low_list, probs_low_list)
                else:
                    plate = img[int(box[1]):int(box[3]),int(box[0]): int(box[2])]
                    # plate = rotate_license_plate(plate)
                    if plate.shape[0] > 0 and plate.shape[1] > 0:
                        _, results = perform_detection(image_path=plate)
                        for label, confidence, bbox in results:
                            x, y, w, h = bbox
                            conf= round(float(confidence),2)
                            index_char.append(label)
                            conf_char.append(conf)
                            box_char.append(bbox)

                        for box, label, conf in zip(box_char, index_char, conf_char):
                            x1, y1, x2, y2 = box
                            cv2.rectangle(plate, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
                        up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                        add_character(label_list['recognition_text'], up_list, label_up_list)
                        add_probs(label_list['confidence'], up_list, probs_up_list)

                end_ocr = time.time()
                print("Time OCR: ", (end_ocr -start_ocr))
                label_text_ocr = ''.join(label_list['recognition_text'])
                if len(label_list['confidence']) == 0:
                    continue
                confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)
                
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
                cv2.putText(img, str(label_text_ocr + '-' + str(confidence_text)), (int(box[0]) + 20, int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            cv2.imwrite(f'ssd/data_evaluate/detection/{image_name}', img)
        else:
            cv2.imwrite(f'ssd/data_evaluate/no_detection/{image_name}', img)
        end_image = time.time()
        print("Time process: ", (end_image - start_image))
    end = time.time()
    print("Average time per frame:", (end-start)/len(list_image))

test_yolo_process('ssd/data_test_3')