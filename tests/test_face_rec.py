import os
import time
import cv2
from models.face_recognition.handler import OpenCVFaceRecognition
from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet import ResNetModel
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def get_correct_bounding_boxes(directory, text_file):
    bounding_boxes = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            with open(text_file, 'r') as file:
                lines = file.readlines()
                for i in range(len(lines)):
                    if lines[i].strip().split('/')[-1] == filename:
                        num_boxes = int(lines[i+1].strip())
                        boxes = []
                        for j in range(num_boxes):
                            x1, y1, w, h, *_ = map(int, lines[i+2+j].strip().split(' '))
                            boxes.append((x1, y1, w, h))
                        bounding_boxes[filename] = boxes
    return bounding_boxes

def count_images_with_faces(face_recognition, directory):
    count = 0
    total = 0
    faces_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            total += 1
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            faces = face_recognition.model.detect_faces(img)
            if len(faces) > 0:
                count += 1
                faces_dict[filename] = faces
    return faces_dict, count, total

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(xi2-xi1, 0) * max(yi2-yi1, 0)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

def calculate_precision_recall(predicted_boxes, correct_boxes):
    tp = 0
    fp = 0
    fn = 0
    for pred_box in predicted_boxes:
        for correct_box in correct_boxes:
            iou = calculate_iou(pred_box, correct_box)
            if iou > 0.5:
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = len(correct_boxes) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall

def calculate_average_precision(precision, recall):
    precision = sorted(precision)
    recall = sorted(recall)
    prev_recall = 0
    ap = 0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap

def calculate_map(faces_dict, bounding_boxes):
    precisions = []
    recalls = []
    for filename in faces_dict:
        predicted_boxes = faces_dict[filename]
        correct_boxes = bounding_boxes[filename]
        precision, recall = calculate_precision_recall(predicted_boxes, correct_boxes)
        precisions.append(precision)
        recalls.append(recall)
    map = calculate_average_precision(precisions, recalls)
    return map

bounding_boxes = get_correct_bounding_boxes('./tests/data/faces/', './tests/data/wider_face_val_bbx_gt.txt')

haar_fast = HaarFrontModel()
haar_fast_handler = OpenCVFaceRecognition(haar_fast)
haar_slow = HaarProfileModel()
haar_slow_handler = OpenCVFaceRecognition(haar_slow)
haar_combined = HaarCombinedModel()
haar_combined_handler = OpenCVFaceRecognition(haar_combined)

resnet = ResNetModel()
resnet_handler = OpenCVFaceRecognition(resnet)

models = [haar_fast_handler, haar_slow_handler, haar_combined_handler, resnet_handler]

faces_dir = './tests/data/faces/'
other_dir = './tests/data/other/'

for model in models:
    print(f"Model: {type(model.model).__name__}")
    start_time = time.time()
    faces_dict, faces_count, total_faces = count_images_with_faces(model, faces_dir)

    

    _, other_count, total_other = count_images_with_faces(model, other_dir)
    time_taken = time.time() - start_time


    total_images = total_faces + total_other
    ap = calculate_map(faces_dict, bounding_boxes)
    print(f"Average Precision: {ap} for detecting faces in the 'faces' directory.")
    print(f"Faces detected in {faces_count / total_faces * 100}% of images from the 'faces' directory.")
    print(f"Faces detected in {other_count / total_other * 100}% of images from the 'other' directory.")
    print(f"Time taken to test {total_images} images: {time_taken} seconds ({total_images/time_taken} images/second)")