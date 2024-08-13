
from ultralytics import YOLO
import cv2
import torch

model = YOLO('ssd/models/ocr/yolo_ocr_recognition.pt')

# img_path = ''

# img = cv2.imread(img_path)

def recognition_charv1(img):
    results = model(img)

    for result in results:

        for bbox in result.boxes:

            index_char = bbox[0].cls
            conf_char = bbox[0].conf
            box_char = bbox[0].xyxy[0]

            (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = split_list(box_char, index_char, conf_char)

            up_list, label_up_list, probs_up_list = arrange_correspond(up_list, label_up_list, probs_up_list)
            low_list, label_low_list, probs_low_list = arrange_correspond(low_list, label_low_list, probs_low_list)

    return (low_list, label_low_list, probs_low_list), (up_list, label_up_list, probs_up_list)

def recognition_charv2(index_char, conf_char, box_char):

    (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = split_list(box_char, index_char, conf_char)

    up_list, label_up_list, probs_up_list = arrange_correspond(up_list, label_up_list, probs_up_list)
    low_list, label_low_list, probs_low_list = arrange_correspond(low_list, label_low_list, probs_low_list)

    return (low_list, label_low_list, probs_low_list), (up_list, label_up_list, probs_up_list)

def split_list(list_boxes, result, probs):
    # Tìm giá trị lớn nhất và nhỏ nhất của phần tử đầu tiên trong các phần tử của danh_sach

    if isinstance(list_boxes, torch.Tensor):
        list_boxes = list_boxes.tolist()
    # print("List Boxes: ",list_boxes)
    if list_boxes:
        max_val = max(list_boxes, key=lambda x: x[1])[1]
        max_val_2 = max(list_boxes, key=lambda x: x[3])[3]
        min_val = min(list_boxes, key=lambda x: x[1])[1]
    else:
        max_val = 0
        max_val_2 = 0
        min_val = 0
    # print("Max val: ", max_val)
    # print("Max val 2: ", max_val_2)
    # print("Min val: ", min_val)
    # Tính giá trị trung bình của hiệu giữa max và min
    tb = (max_val_2 - min_val) / 2
    print('Trung Bình: ',tb)
    # Khởi tạo các danh sách mới
    list_boxes_1, result_1, probs_1 = [], [], []
    list_boxes_2, result_2, probs_2 = [], [], []
    
    # Phân chia danh_sach, result, và probs thành hai nhóm
    for ds, res, prob in zip(list_boxes, result, probs):
        if ds[1] > tb:
            list_boxes_1.append(ds)
            result_1.append(res)
            probs_1.append(prob)
        else:
            list_boxes_2.append(ds)
            result_2.append(res)
            probs_2.append(prob)
    
    return (list_boxes_1, result_1, probs_1), (list_boxes_2, result_2, probs_2)

def arrange_correspond(list_boxes, result, probs):
    # Kết hợp các phần tử của danh_sach, result, và probs thành một danh sách duy nhất
    combined = list(zip(list_boxes, result, probs))
    
    # Sắp xếp danh sách kết hợp này dựa trên các phần tử trong danh_sach
    combined_sorted = sorted(combined, key=lambda x: x[0][0])
    if not combined_sorted:
        return [], [], []
    # Tách danh sách kết hợp thành ba danh sách riêng biệt
    danh_sach_sorted, result_sorted, probs_sorted = zip(*combined_sorted)
    
    return list(danh_sach_sorted), list(result_sorted), list(probs_sorted)
