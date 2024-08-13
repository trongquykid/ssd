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

def test_image(image_path):
    list_image = os.listdir(image_path)
    for image_name in list_image:
        image = cv2.imread(f"{image_path}/{image_name}")
        height, width = image.shape[:2]
        aspect_ratio = width / height
        label_list = {'recognition_text' : [],
                        'confidence' : []}
        print('Aspect ratio: ',aspect_ratio)
        if 0 <= aspect_ratio <= 2.0:
            # image_rotate, flag = rotate_license_plate(image)
            # if flag:
            #     cv2.imwrite(f'./test_ocr/_rotate_{image_name}', image_rotate)
            # cv2.imwrite(f"image_rotate/{image_name}", image_rotate)
            results = model_ocr(image, conf = 0.4)
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    index_char = result.boxes.cls
                    conf_char = result.boxes.conf
                    box_char = result.boxes.xyxy

                    (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list)= recognition_charv2(index_char, conf_char, box_char)

                    add_character(label_list['recognition_text'], up_list, label_up_list)
                    add_probs(label_list['confidence'], up_list, probs_up_list)

                    add_character(label_list['recognition_text'], low_list, label_low_list)
                    add_probs(label_list['confidence'], low_list, probs_low_list)
                    
        else:
            # image_rotate, flag = rotate_license_plate(image)
            # if flag:
            #     cv2.imwrite(f'./test_ocr/_rotate_{image_name}', image_rotate)
        #     cv2.imwrite(f"image_rotate/{image_name}", image_rotate)
            results = model_ocr(image, conf = 0.5)
            
            for result in results:
                index_char = result.boxes.cls
                conf_char = result.boxes.conf
                box_char = result.boxes.xyxy
        
                up_list, label_up_list, probs_up_list = arrange_correspond(box_char, index_char, conf_char)

                add_character(label_list['recognition_text'], up_list, label_up_list)
                add_probs(label_list['confidence'], up_list, probs_up_list)
        
        label_text_ocr = ''.join(label_list['recognition_text'])
        if len(label_list['confidence']) == 0:
            continue
        confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)
        output_path = f"./test_ocr/_{label_text_ocr}_{image_name}"
        cv2.imwrite(output_path, image)

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
                    if plate.shape[0] > 0 and plate.shape[1] > 0:
                        results = model_ocr(plate, conf=0.6)

                        for result in results:

                            index_char = result.boxes.cls
                            conf_char = result.boxes.conf
                            box_char = result.boxes.xyxy
                            print(conf_char)
                            print(index_char)
                            print(box_char)
                            for box, index, conf in zip(box_char, index_char, conf_char):
                                x1, y1, x2, y2 = box
                                label_ocr = class_name_ocr[int(index)]
                                conf= round(float(conf),2)
                                label_str = f'{label_ocr}:{conf}'
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
                        results = model_ocr(plate, conf=0.6)

                        for result in results:

                            index_char = result.boxes.cls
                            conf_char = result.boxes.conf
                            box_char = result.boxes.xyxy
                            for box in box_char:
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

def test_ocr(image): 
    results = model_ocr(image, conf=0.6)
    label_list = {'recognition_text' : [],'confidence' : []}
    for result in results:

        index_char = result.boxes.cls
        conf_char = result.boxes.conf
        box_char = result.boxes.xyxy
        # print(conf_char)
        # print(index_char)
        # print(box_char)
        for box, index, conf in zip(box_char, index_char, conf_char):
            x1, y1, x2, y2 = box
            label_ocr = class_name_ocr[int(index)]
            conf= round(float(conf),2)
            label_str = f'{label_ocr}:{conf}'
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)
            # cv2.putText(image, str(label_str),
            # (int(x1), int(y1) - 10),
            # cv2.FONT_HERSHEY_SIMPLEX,
            # 0.6,  # font scale
            # (255, 0, 255),
            # 2)  # line type
        (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list)= recognition_charv2(index_char, conf_char, box_char)

        add_character(label_list['recognition_text'], up_list, label_up_list)
        add_probs(label_list['confidence'], up_list, probs_up_list)

        add_character(label_list['recognition_text'], low_list, label_low_list)
        add_probs(label_list['confidence'], low_list, probs_low_list)

    label_text_ocr = ''.join(label_list['recognition_text'])
    # confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)
    print("Text Plate: ",label_text_ocr)

def process_frame(frame):
    boxes, labels, probs = predictor.predict(frame, 10, 0.4)

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

        label_list = {'recognition_text' : [],'confidence' : []}

        if 0 <= aspect_ratio <= 2.0:
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
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
            plate = frame[int(box[1]):int(box[3]), int(box[0]): int(box[2])]
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
            confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)
        else:
            confidence_text = 0

        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        # label_text = f"{class_names[label]}: {score:.2f}"
        cv2.putText(frame, str(label_text_ocr + '-' + str(confidence_text)), (int(box[0]) + 20, int(box[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    return frame

def process_videov2(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    tracker = Tracker()
    frame_number = 0

    preds = []
    # pre_time = time.time()
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame
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
                label_list = {'recognition_text' : [],'confidence' : []}
                
                if 0 <= aspect_ratio <= 2.5:
                    plate = frame[int(y1):int(y2), int(x1): int(x2)]

                    plate = rotate_license_plate(plate)
                    
                    if plate is None:
                        continue

                    # plate = cv2.GaussianBlur(plate, (3, 3), 0)
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
                    plate = frame[int(y1):int(y2), int(x1): int(x2)]

                    plate = rotate_license_plate(plate)
                    # plate = rotate_license_plate(plate)
                    # plate = cv2.GaussianBlur(plate, (3, 3), 0)
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
                    confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)

                output_frame = {"track_id": track_id, "ocr_txt": label_text_ocr, "ocr_conf": confidence_text}
                preds.append(output_frame)

                if track_id in list(set(ele['track_id'] for ele in preds)):
                    rec_conf, ocr_resc = get_best_ocr(preds, track_id)
                    save_info_file(f"./save_file_txt", 'log', [{'Track_id': track_id, 'Recognized_text': ocr_resc, 'Confidence': rec_conf, 'Aspect_ratio' : aspect_ratio}])
                txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
                
        else:
            continue

        out.write(frame)

        frame_number += 1
        print(f'Processing frame {frame_number}/{total_frames}', end='\r')
    after_time = time.time()
    # print("Total time: ", (after_time - start))
    
    cap.release()
    out.release()
    print("\nProcessing complete.")

def process_videov2_1(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    tracker = Tracker()
    frame_number = 0
    avg_fps = 0
    preds = []
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        pre_time = time.time()
        if not ret:
            break
        
        # Process each frame
        boxes, labels, probs = predictor.predict(frame, 10, 0.4)
        result_tracker = []

        for i in range(boxes.size(0)):
            box = boxes[i]
            score = probs[i]
            x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            result_tracker.append([x_min, y_min, x_max, y_max, score])

        tracker.update(frame, result_tracker)

        if boxes is not None and len(boxes) > 0:
            try:
                for track in tracker.tracks:
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.bbox
                    width = x2 - x1
                    height = y2 - y1

                    # Check for valid box dimensions
                    if width <= 0 or height <= 0:
                        continue

                    aspect_ratio = width / height
                    label_list = {'recognition_text' : [],'confidence' : []}
                    
                    if 0 <= aspect_ratio <= 2.5:
                        plate = frame[int(y1):int(y2), int(x1): int(x2)]

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
                        plate = frame[int(y1):int(y2), int(x1): int(x2)]

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
                        confidence_text = round(sum(label_list['confidence'])/len(label_list['confidence']), 2)

                    output_frame = {"track_id": track_id, "ocr_txt": label_text_ocr, "ocr_conf": confidence_text}
                    preds.append(output_frame)

                    if track_id in list(set(ele['track_id'] for ele in preds)):
                        rec_conf, ocr_resc = get_best_ocr(preds, track_id)
                        save_info_file(f"./save_file_txt", 'log', [{'Track_id': track_id, 'Recognized_text': ocr_resc, 'Confidence': rec_conf, 'Aspect_ratio' : aspect_ratio}])
                        txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)

                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(frame, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        continue
            except Exception as e:
                print(f"Error processing track: {e}")
                continue
        tot_time = time.time() - pre_time
        
        fps = round(1 / tot_time,2)
        avg_fps += fps
        # Writing information onto the frame and saving it to be processed in a video.
        cv2.putText(frame, 'frame: %d fps: %s' % (frame_number, fps),
                    (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
        out.write(frame)
        frame_number += 1
        print(f'Processing frame {frame_number}/{total_frames}', end='\r')
    print("Avg FPS: ", avg_fps / total_frames)
    print("Total time: ", time.time() - start)
    cap.release()
    out.release()
    print("\nProcessing complete.")

def process_videov2_2(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    tracker = Tracker()
    frame_number = 0

    preds = []
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        
        pre_time = time.time()
        if ret == True:
            
            overlay_img = frame.copy()
            # Process each frame
            boxes, labels, probs = predictor.predict(frame, 10, 0.4)
            result_tracker = []

            for i in range(boxes.size(0)):
                box = boxes[i]
                score = probs[i]
                x_min, y_min, x_max, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                result_tracker.append([x_min, y_min, x_max, y_max, score])

            tracker.update(overlay_img, result_tracker)

            if result_tracker:
                for track in tracker.tracks:
                    track_id = track.track_id
                    x1, y1, x2, y2 = track.bbox
                    width = x2 - x1
                    height = y2 - y1

                    # Check for valid box dimensions
                    if width <= 0 or height <= 0:
                        continue

                    aspect_ratio = width / height
                    label_list = {'recognition_text' : [], 'confidence' : []}

                    plate = frame[int(y1):int(y2), int(x1): int(x2)]
                    plate = rotate_license_plate(plate)

                    if plate is None or plate.size == 0:  # Check if the plate is empty or None
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

                    label_text_ocr = ''.join(label_list['recognition_text'])
                    confidence_text = round(sum(label_list['confidence']) / len(label_list['confidence']), 2) if label_list['confidence'] else 0

                    output_frame = {"track_id": track_id, "ocr_txt": label_text_ocr, "ocr_conf": confidence_text}
                    preds.append(output_frame)

                    if track_id in list(set(ele['track_id'] for ele in preds)):
                        rec_conf, ocr_resc = get_best_ocr(preds, track_id)
                        save_info_file(f"./save_file_txt", 'log', [{'Track_id': track_id, 'Recognized_text': ocr_resc, 'Confidence': rec_conf, 'Aspect_ratio' : aspect_ratio}])
                        txt = str(track_id) + ": " + str(ocr_resc) + '-' + str(rec_conf)

                        cv2.rectangle(overlay_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        cv2.putText(overlay_img, txt, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        continue

            tot_time = time.time() - pre_time
            fps = round(1 / tot_time, 2)

            # Writing information onto the frame and saving it to be processed in a video.
            cv2.putText(overlay_img, f'frame: {frame_number} fps: {fps}', (0, int(100 * 1)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
            out.write(overlay_img)
            frame_number += 1
            print(f'Processing frame {frame_number}/{total_frames}', end='\r')
        else:
            break

    print("\nProcessing complete.")

def process_video_v3(output_video_path):
    cap = cv2.VideoCapture(0)  # 0 is the ID for the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    # out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    tracker = Tracker()
    frame_number = 0

    preds = []

    while cap.isOpened():
        prev_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        img_tmp = frame
        # Process each frame
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
        tot_time = time.time() - prev_time
        fps = round(1 / tot_time, 2)
        cv2.putText(img_tmp, 'frame: %d fps: %s' % (frame_number, fps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Processed Frame', frame)
        # out.write(frame)

        frame_number += 1
        print(f'Processing frame {frame_number}', end='\r')

    cap.release()
    # out.release()
    print("\nProcessing complete.")

def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        prev_time = time.time()
        img_tmp = frame

        processed_frame = process_frame(frame)
        out.write(processed_frame)

        frame_number += 1
        print(f'Processing frame {frame_number}/{total_frames}', end='\r')
    
    cap.release()
    out.release()
    print("\nProcessing complete.")

# Example usage
# process_video('input_video.mp4', 'output_video.avi', predictor, model_ocr, recognition_charv2, class_name_ocr)


# Example usage
# process_video('input_video.mp4', 'output_video.avi', predictor, model_ocr, recognition_charv2, class_name_ocr)

# Lưu ảnh đầu ra
INPUT_DIR = 'test_video/test_19_7mp4.mp4' 
OUT_PATH = './results/out_test_19_7mp4.mp4'

process_videov2_1(INPUT_DIR, OUT_PATH)

# process_video(INPUT_DIR, OUT_PATH)

# process_frame_deepsort(INPUT_DIR, OUT_PATH)

test_yolo_process('ssd/data_test_3')

# test_image("./test_ocr")

# roate_image = rotate_license_plate(image)

# cv2.imwrite('./test_ocr/roate_image.jpg', roate_image)

# Example usage
# process_video_v3("output_video.avi")

# test_ocr("./rotate_plate/frame_1533.2202042255376.jpg")