import os
import time
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from models.face_recognition.handler import OpenCVFaceHandler
from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet_caffee import ResNetCaffeeModel
from models.face_recognition.yunet import YuNetModel


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

def get_classification_labels(predicted_boxes, correct_boxes):
    y_true, y_pred = [], []

    matched = set()
    for pred_box in predicted_boxes:
        best_iou, best_match = max([(calculate_iou(pred_box, correct_box), idx) for idx, correct_box in enumerate(correct_boxes)], default=(0, -1))
        if best_iou > 0.5 and best_match not in matched:
            y_true.append(1)  # True Positive
            matched.add(best_match)
        else:
            y_true.append(0)  # False Positive
        y_pred.append(1)

    # Add False Negatives
    y_true.extend([1] * (len(correct_boxes) - len(matched)))
    y_pred.extend([0] * (len(correct_boxes) - len(matched)))

    return y_true, y_pred

def calculate_metrics_for_image(predicted_boxes, correct_boxes):
    tp = fp = fn = 0

    matched = set()

    for pred_box in predicted_boxes:
        iou_scores = [calculate_iou(pred_box, correct_box) for correct_box in correct_boxes]
        max_iou = max(iou_scores, default=0)
        if max_iou > 0.5:
            tp += 1
            matched.add(iou_scores.index(max_iou))
        else:
            fp += 1

    fn = len(correct_boxes) - len(matched)
    return tp, fp, fn


def evaluate_model(model, directory, bounding_boxes):
    all_y_true, all_y_pred = [], []
    tp, fp, fn = 0, 0, 0
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            predicted_boxes = model.detect_faces(img)
            correct_boxes = bounding_boxes.get(filename, [])

            ttp, ffp, ffn = calculate_metrics_for_image(predicted_boxes, correct_boxes)
            tp += ttp
            fp += ffp
            fn += ffn

            y_true, y_pred = get_classification_labels(predicted_boxes, correct_boxes)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    print(f"true positives: {tp}, false positives:{fp}, false negatives:{fn}")

    return precision, recall, f1

faces_dir = './tests/data/face_detection/'
bounding_boxes = get_correct_bounding_boxes(faces_dir, './tests/data/wider_face_val_bbx_gt.txt')

haar = HaarFrontModel()
#haar_handler = OpenCVFaceHandler(haar)

resnet_caffee = ResNetCaffeeModel()
# resnet_caffee_handler = OpenCVFaceHandler(resnet_caffee)

yunet = YuNetModel()
# yunet_handler = OpenCVFaceHandler(yunet)

models = [haar, resnet_caffee, yunet]


for model in models:
    print(f"Model: {type(model).__name__}")
    start_time = time.time()

    precision, recall, f1 = evaluate_model(model, faces_dir, bounding_boxes)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken} seconds")

# print("YuNet with different thresholds")
# for threshold in np.arange(0.1, 1.0, 0.1):
#     yunet_handler.model.detector.setScoreThreshold(threshold)
#     start_time = time.time()
#     precision, recall, f1 = evaluate_model(yunet_handler, faces_dir, bounding_boxes)
#     time_taken = time.time() - start_time
#     print(f"Threshold: {threshold} Precision: {precision} Recall: {recall} F1-Score: {f1} Time taken: {time_taken}")

# print("ResNet with different thresholds")
# for threshold in np.arange(0.1, 1.0, 0.1):
#     resnet_caffee_handler.model.confidence_threshold = threshold
#     start_time = time.time()
#     precision, recall, f1 = evaluate_model(resnet_caffee_handler, faces_dir, bounding_boxes)
#     time_taken = time.time() - start_time
#     print(f"Threshold: {threshold} Precision: {precision} Recall: {recall} F1-Score: {f1} Time taken: {time_taken}")

# Initialize lists to store results
yunet_precision, yunet_recall, yunet_f1, yunet_time = [], [], [], []
resnet_precision, resnet_recall, resnet_f1, resnet_time = [], [], [], []

thresholds = np.arange(0.1, 1.0, 0.1)

# YuNet
for threshold in thresholds:
    yunet.detector.setScoreThreshold(threshold)
    start_time = time.time()
    precision, recall, f1 = evaluate_model(yunet, faces_dir, bounding_boxes)
    time_taken = time.time() - start_time
    yunet_precision.append(precision)
    yunet_recall.append(recall)
    yunet_f1.append(f1)
    yunet_time.append(time_taken)

# ResNet
for threshold in thresholds:
    resnet_caffee.confidence_threshold = threshold
    start_time = time.time()
    precision, recall, f1 = evaluate_model(resnet_caffee, faces_dir, bounding_boxes)
    time_taken = time.time() - start_time
    resnet_precision.append(precision)
    resnet_recall.append(recall)
    resnet_f1.append(f1)
    resnet_time.append(time_taken)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(thresholds, yunet_precision, label='YuNet')
plt.plot(thresholds, resnet_precision, label='ResNet')
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(thresholds, yunet_recall, label='YuNet')
plt.plot(thresholds, resnet_recall, label='ResNet')
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(thresholds, yunet_f1, label='YuNet')
plt.plot(thresholds, resnet_f1, label='ResNet')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(thresholds, yunet_time, label='YuNet')
plt.plot(thresholds, resnet_time, label='ResNet')
plt.xlabel('Threshold')
plt.ylabel('Time Taken (seconds)')
plt.legend()

plt.tight_layout()
plt.show()