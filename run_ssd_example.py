from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import onnxruntime as ort
import os
import random
import time


net_type = 'mb2-ssd-lite'
model_path = 'models/plate/az_plate/plate_v1_ssdmobilenetv2.onnx'
label_path = 'models/plate/az_plate/labels.txt'
image_path = 'ImageQuy_1.jpg'


class_names = [name.strip() for name in open(label_path).readlines()]

# if net_type == 'vgg16-ssd':
#     net = create_vgg_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd':
#     net = create_mobilenetv1_ssd(len(class_names), is_test=True)
# elif net_type == 'mb1-ssd-lite':
#     net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'mb2-ssd-lite':
#     net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
# elif net_type == 'sq-ssd-lite':
#     net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
# else:
#     print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
# net = create_vgg_ssd(len(class_names), is_test=True)
#     # sys.exit(1)
# net.load(model_path)

session = ort.InferenceSession(model_path)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(session, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(session, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_predictor(session, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(session, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(session, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(session, candidate_size=200)

def random_crop(image, crop_fraction=0.1):
    """
    Crop a small portion of the image randomly.
    
    Parameters:
    - image: Input image to be cropped.
    - crop_fraction: Fraction of the image size to be cropped. Default is 0.1 (10%).
    
    Returns:
    - cropped_image: The cropped and resized image.
    """
    original_height, original_width = image.shape[:2]

    # Calculate the maximum pixels to be cropped based on the fraction
    max_crop_height = int(original_height * crop_fraction)
    max_crop_width = int(original_width * crop_fraction)
    
    # Ensure the crop is symmetric around the center
    x_left = random.randint(0, max_crop_width)
    x_right = original_width - random.randint(0, max_crop_width)
    
    y_top = random.randint(0, max_crop_height)
    y_bottom = original_height - random.randint(0, max_crop_height)
    
    # Crop the image with calculated coordinates
    cropped_image = image[y_top:y_bottom, x_left:x_right]
    # Resize the cropped image to the original size
    cropped_image = cv2.resize(cropped_image, (original_width, original_height))

    return cropped_image

def detect_plate(path_image):
    
    list_image = os.listdir(path_image)
    start = time.time()
    for image_name in list_image:
        image_path = f"{path_image}/{image_name}"
        orig_image = cv2.imread(image_path)
        # zoom_factor = 2.5  # Change this factor to zoom in more or less
        # image = cv2.resize(orig_image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        # cv2.imwrite(f'./data_test_2/new_image/zoom_{image_name}', image)

        # gray_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        # img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
        # img = cv2.equalizeHist(img)
        # image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # image = cv2.GaussianBlur(orig_image, (3, 3), 0)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        # image = cv2.GaussianBlur(orig_image, (3, 3), 0)
        # print(image_name)
        # image = random_crop(image)
        # cv2.imwrite(f'./data_test_2/new_image/zoom_{image_name}', image)
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        
        if len(boxes) == 0:
            path = f"./data_ocr_evaluate/results/test_{image_name}"
            cv2.imwrite(path, image)
        else:
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = width / height

                # if 0 < aspect_ratio <= 2.0:
                #     path_2_line = f"./data_test_2/two_line/{image_name}"
                #     cv2.imwrite(path_2_line, image)
                # else:
                #     path_1_line = f"./data_test_2/one_line/{image_name}"
                #     cv2.imwrite(path_1_line, image)

                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 6)
                #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
                label = f"{class_names[labels[i]-1]}: {probs[i]:.2f}"
                cv2.putText(image, label,
                            (int(box[0]), int(box[1]) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            3,  # font scale
                            (255, 0, 255),
                            10)  # line type

            path = f"./data_ocr_evaluate/results/test_{image_name}"
            cv2.imwrite(path, image)
    end = time.time()
    print("Average time per frame:", (end-start)/len(list_image))

def detect_ocr(path_image):
    
    list_image = os.listdir(path_image)
    for image_name in list_image:
        image_path = f"{path_image}/{image_name}"
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        cv2.imshow(image, "image")
        # image = cv2.resize(image, (1024, 640))
        boxes, labels, probs = predictor.predict(image, 10, 0.0)
        if len(boxes) == 0:
            path = f"./data_evaluate/{image_name}"
            cv2.imwrite(path, orig_image)
        else:
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
                #label = f"""{voc_dataset.class_names[labels[i]]}: {probs[i]:.2f}"""
                label = f"{class_names[labels[i]-1]}: {probs[i]:.2f}"
                cv2.putText(orig_image, label,
                            (int(box[0]) + 20, int(box[1]) + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type

            path = f"./data_evaluate/{image_name}"
            cv2.imwrite(path, orig_image)

detect_plate('data_ocr_evaluate')
