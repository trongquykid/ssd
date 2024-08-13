import torch
from ..utils import box_utils
from .data_preprocessing import PredictionTransform
from ..utils.misc import Timer
import onnxruntime as ort  # Make sure to import ONNX Runtime

class Predictor:
    def __init__(self, net, size, mean=0.0, std=1.0, nms_method=None,
                 iou_threshold=0.45, filter_threshold=0.01, candidate_size=200, sigma=0.5, device=None):
        self.net = net
        self.transform = PredictionTransform(size, mean, std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.nms_method = nms_method
        self.sigma = sigma
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # If net is a PyTorch model, move it to the appropriate device
        if isinstance(self.net, torch.nn.Module):
            self.net.to(self.device)
            self.net.eval()

        self.timer = Timer()

    def predict(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        # height, width = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        
        # Move images to device if net is a PyTorch model
        if isinstance(self.net, torch.nn.Module):
            images = images.to(self.device)
        
        with torch.no_grad():
            self.timer.start()
            if isinstance(self.net, torch.nn.Module):
                scores, boxes = self.net.forward(images)
            elif isinstance(self.net, ort.InferenceSession):
                ort_inputs = {self.net.get_inputs()[0].name: images.cpu().numpy()}
                ort_outs = self.net.run(None, ort_inputs)
                scores, boxes = torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1])
            else:
                raise TypeError("Unsupported model type")
            print("Inference time: ", self.timer.end())
        
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        
        # This version of NMS is slower on GPU, so we move data to CPU
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):

            probs = scores[:, class_index]

            mask = probs > prob_threshold
            probs = probs[mask]
            # print(probs)
            if probs.size(0) == 0:
                continue
            # print(class_index)
            subset_boxes = boxes[mask, :]
            
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
            # picked_labels.extend([class_index] * len(box_probs))
        # print(picked_labels)
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]
    
    def predict_character(self, image, top_k=-1, prob_threshold=None):
        cpu_device = torch.device("cpu")
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0)
        
        # Move images to device if net is a PyTorch model
        if isinstance(self.net, torch.nn.Module):
            images = images.to(self.device)
        
        with torch.no_grad():
            self.timer.start()
            if isinstance(self.net, torch.nn.Module):
                scores, boxes = self.net.forward(images)
            elif isinstance(self.net, ort.InferenceSession):
                ort_inputs = {self.net.get_inputs()[0].name: images.cpu().numpy()}
                ort_outs = self.net.run(None, ort_inputs)
                scores, boxes = torch.tensor(ort_outs[0]), torch.tensor(ort_outs[1])
            else:
                raise TypeError("Unsupported model type")
            print("Inference time: ", self.timer.end())
        
        boxes = boxes[0]
        scores = scores[0]
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        
        # This version of NMS is slower on GPU, so we move data to CPU
        boxes = boxes.to(cpu_device)
        scores = scores.to(cpu_device)

        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):

            probs = scores[:, class_index]

            mask = probs > prob_threshold
            probs = probs[mask]
            # print(probs)
            if probs.size(0) == 0:
                continue
            # print(class_index)
            subset_boxes = boxes[mask, :]
            
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            
            box_probs = box_utils.nms(box_probs, self.nms_method,
                                      score_threshold=prob_threshold,
                                      iou_threshold=self.iou_threshold,
                                      sigma=self.sigma,
                                      top_k=top_k,
                                      candidate_size=self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
            # picked_labels.extend([class_index] * len(box_probs))
        # print(picked_labels)
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        (up_list, label_up_list, probs_up_list), (low_list, label_low_list, probs_low_list) = split_list(list(picked_box_probs[:, :4]), list(torch.tensor(picked_labels)), list(picked_box_probs[:, 4]))
        
        print(up_list, label_up_list, probs_up_list)
        print("------------")
        print(low_list, label_low_list, probs_low_list)

        up_list, label_up_list, probs_up_list = arrange_correspond(up_list, label_up_list, probs_up_list)
        low_list, label_low_list, probs_low_list = arrange_correspond(low_list, label_low_list, probs_low_list)

        return (low_list, label_low_list, probs_low_list), (up_list, label_up_list, probs_up_list)
    
def split_list(list_boxes, result, probs):
    # Tìm giá trị lớn nhất và nhỏ nhất của phần tử đầu tiên trong các phần tử của danh_sach
    max_val = max(list_boxes, key=lambda x: x[1])[1]
    min_val = min(list_boxes, key=lambda x: x[1])[1]
    
    print(max_val)
    print(min_val)

    # Tính giá trị trung bình của hiệu giữa max và min
    tb = (max_val - min_val) / 2
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
    # Tách danh sách kết hợp thành ba danh sách riêng biệt
    danh_sach_sorted, result_sorted, probs_sorted = zip(*combined_sorted)
    
    return list(danh_sach_sorted), list(result_sorted), list(probs_sorted)
