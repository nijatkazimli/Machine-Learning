import os
import time
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score

from models.face_recognition.handler import OpenCVFaceRecognition
from models.face_recognition.haar import HaarFrontModel, HaarProfileModel, HaarCombinedModel
from models.face_recognition.resnet import ResNetModel

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


def evaluate_model(face_recognition, directory, bounding_boxes):
    all_y_true, all_y_pred = [], []

    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png")):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            predicted_boxes = face_recognition.model.detect_faces(img)
            correct_boxes = bounding_boxes.get(filename, [])

            y_true, y_pred = get_classification_labels(predicted_boxes, correct_boxes)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)

    return precision, recall, f1

bounding_boxes = get_correct_bounding_boxes('./data/faces/', './data/wider_face_val_bbx_gt.txt')

haar_fast = HaarFrontModel()
haar_fast_handler = OpenCVFaceRecognition(haar_fast)
haar_slow = HaarProfileModel()
haar_slow_handler = OpenCVFaceRecognition(haar_slow)
haar_combined = HaarCombinedModel()
haar_combined_handler = OpenCVFaceRecognition(haar_combined)

resnet = ResNetModel()
resnet_handler = OpenCVFaceRecognition(resnet)

models = [haar_fast_handler, haar_slow_handler, haar_combined_handler, resnet_handler]

faces_dir = './data/faces/'
other_dir = './data/other/'

for model in models:
    print(f"Model: {type(model.model).__name__}")
    start_time = time.time()

    precision, recall, f1 = evaluate_model(model, faces_dir, bounding_boxes)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")

    time_taken = time.time() - start_time
    print(f"Time taken: {time_taken} seconds")