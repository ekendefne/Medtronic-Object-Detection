import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import warnings
import pybboxes as pbx
import seaborn as sns

# Path to the COCO validation dataset
coco_data_path = Path("data")

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)

#pre-process
def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	detections = net.forward(output_layers)
	return detections

#post-process
def post_process(input_image, outputs):
	classes = None
	with open("coco.names") as f:
		classes = f.read().strip().split('\n')
	# Lists to hold respective values while unwrapping.
	class_ids = []
	confidences = []
	boxes = []
	# Rows.
	rows = outputs[0].shape[1]
	image_height, image_width = input_image.shape[:2]

	# Resizing factor.
	x_factor = image_width / INPUT_WIDTH
	y_factor =  image_height / INPUT_HEIGHT
	# Iterate through 25200 detections.
	for r in range(rows):
		row = outputs[0][0][r]
		confidence = row[4]
		classes_scores = row[5:]
		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				#turn images into their original size
				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
				box = np.array([left, top, width, height])
				boxes.append(box)
	
	#the predictions are an array in the format:[x1,y1,x2,y2,conf,class]
	predictions = list(zip(boxes, confidences,class_ids))
	cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
	return predictions,input_image

# Load COCO labels
with open("coco.names") as f:
	coco_labels = f.read().strip().split('\n')

def get_labels(image_path):
	image_filename = image_path.stem  # Extract the filename without extension
	image_file_path = coco_data_path / 'frames' / f"{image_filename}.jpg"
	
	image = cv2.imread(str(image_file_path))
	H, W = image.shape[:2]
	
	# Give the weight files to the model and load the network using them.
	modelWeights = "models/yolov5n.onnx"#change accordingly 
	model="yolo5n"
	net = cv2.dnn.readNet(modelWeights)

	# Process image.
	detections = pre_process(image, net)
	predictions,pred_image=post_process(image.copy(),detections)
	
	pred_list = []
	for item in predictions:
	# Extract values from the tuple
		array_values = item[0]
		value1 = item[1]
		value2 = item[2]
		# Flatten the array and create the desired format
		pred_label= tuple(array_values.tolist()) + (value1, value2)
		

		# Add the flattened item to the new list
		pred_list.append(pred_label)
		pred_numpy_array = np.array(pred_list)

	labels_file_path = coco_data_path / 'labels' / f"{image_filename}.txt"
	gt_labels=[]
			
	with open(labels_file_path) as f:
		lines = f.readlines()
		gt_labels = []
		for line in lines:
			values = [float(num) for num in line.split()]
			# Ensure there are exactly 5 numbers in each line
			if len(values) == 5:
				gt_labels.append(values)
		
		gt_list = []
		
		#the txt files in "data/labels" are in [class,x_centre,y_center,width,height] format
		#'pbx.convert_box' converts them to [class,x1,y1,x2,y2] format 
		for array in gt_labels:
			class_label = array[0]
			normalized_coordinates = array[1:]
			
			# YOLO format to OpenCV format
			denormalized_coordinates =pbx.convert_bbox(normalized_coordinates, from_type="yolo", to_type="voc", image_size=(W,H))
			denormalized_array = [class_label] + list(denormalized_coordinates)
			
			# Add the OpenCV values to the list
			gt_list.append((denormalized_array))
			formatted_gt_list = [tuple(map(int, item)) for item in gt_list]
			cv2.rectangle(image, (denormalized_coordinates[0], denormalized_coordinates[1]), (denormalized_coordinates[2], denormalized_coordinates[3]), RED, 3*THICKNESS)
		
	#to see the predictions of the model against ground truth labels uncomment:
			
	# cv2.imshow('Predicted',pred_image )
	# cv2.imshow('Ground Truth',image )	
	# cv2.waitKey(0)
			
	return np.array(formatted_gt_list), pred_numpy_array,model


## Confusion Matrix code modified from: https://github.com/kaanakan/object_detection_confusion_matrix.git ##

def box_iou_calc(boxes1, boxes2):
	# https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
	"""
	Return intersection-over-union (Jaccard index) of boxes.
	Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
	Arguments:
		boxes1 (Array[N, 4])
		boxes2 (Array[M, 4])
	Returns:
		iou (Array[N, M]): the NxM matrix containing the pairwise
			IoU values for every element in boxes1 and boxes2

	This implementation is taken from the above link and changed so that it only uses numpy..
	"""


	def box_area(box):
		# box = 4xn
		return (box[2] - box[0]) * (box[3] - box[1])

	area1 = box_area(boxes1.T)
	area2 = box_area(boxes2.T)

	lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
	rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

	inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
	return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
	def __init__(self, num_classes: int, CONF_THRESHOLD=0.2, IOU_THRESHOLD=0.3):
		self.matrix = np.zeros((num_classes, num_classes))
		self.num_classes = num_classes
		self.CONF_THRESHOLD = CONF_THRESHOLD
		self.IOU_THRESHOLD = IOU_THRESHOLD

	def process_batch(self, detections, labels):

		gt_classes = labels[:, 0].astype(np.int16)

		try:
			detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
		except IndexError or TypeError:
			# detections are empty, end of process
			for i, label in enumerate(labels):
				gt_class = gt_classes[i]
				self.matrix[self.num_classes-1, gt_class] += 1
			return

		detection_classes = detections[:, 5].astype(np.int16)

		for i, label in enumerate(labels):
			gt_class = gt_classes[i]
			# Check if there is any detection with IoU above the threshold
			matching_detections = []

			for j, detection in enumerate(detections):
				iou = box_iou_calc(np.array([label[1:]]), np.array([detection[:4]]))[0]
			
				if iou > self.IOU_THRESHOLD:
					matching_detections.append((j, iou))
			
			#print(f"Ground Truth {i + 1} - Class {gt_class}:")
			#print("Matching Detections:", matching_detections)

			if matching_detections:
				# Sort by IoU and use the one with the highest IoU
				matching_detections.sort(key=lambda x: x[1], reverse=True)
				detection_class = detection_classes[matching_detections[0][0]]
				self.matrix[detection_class, gt_class] += 1
			else:
				# False Negative: Ground truth box is not detected
				self.matrix[self.num_classes-1, gt_class] += 1

		for i, detection in enumerate(detections):
			# Check if the detection has any matching ground truth box
			matching_ground_truths = []

			for j, label in enumerate(labels):
				iou = box_iou_calc(np.array([label[1:]]), np.array([detection[:4]]))[0]
				#print (iou)
				if iou > self.IOU_THRESHOLD:
					matching_ground_truths.append((j, iou))

			#print(f"Detection {i + 1} - Class {detection_classes[i]}:")
			#print("Matching Ground Truths:", matching_ground_truths)

			if not matching_ground_truths:
				# False Positive: Predicted box has no matching ground truth
				detection_class = detection_classes[i]
				self.matrix[detection_class, self.num_classes-1] += 1

	def return_matrix(self):
		return self.matrix


	def plot(self, normalize=True, names=(),image_name=str,model=str):
			import seaborn as sn

			array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9) if normalize else 1)  # normalize columns
			array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

			fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
			nc, nn = self.num_classes, len(names)  # number of classes, names
			sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
			labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
			ticklabels = (names + ['background']) if labels else 'auto'
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
				sn.heatmap(array,
						ax=ax,
						annot=nc < 30,
						annot_kws={
							'size': 8},
						cmap='Blues',
						fmt='.2f',
						square=True,
						vmin=0.0,
						xticklabels=ticklabels,
						yticklabels=ticklabels).set_facecolor((1, 1, 1))
			ax.set_xlabel('True')
			ax.set_ylabel('Predicted')
			ax.set_title(f'Confusion Matrix of {model} - {image_name}')
			plt.show()
			plt.close(fig)

#calculats average precision and recall for each image
def pre_rec(confusion_matrix):
	cm = np.array(confusion_matrix)
	true_pos = np.diag(cm)
	false_pos = np.sum(cm, axis=0) - true_pos
	false_neg = np.sum(cm, axis=1) - true_pos

	# print("True Positives:", true_pos)
	# print("False Positives:", false_pos)
	# print("False Negatives:", false_neg)

	precision = true_pos / np.maximum(true_pos + false_pos, 1e-12)  # Avoid division by zero
	recall = true_pos / np.maximum(true_pos + false_neg, 1e-12)  # Avoid division by zero

	# print("Precision:", precision)
	# print("Recall:", recall)

	average_precision = np.nanmean(precision)
	average_recall = np.nanmean(recall)

	return average_precision,average_recall



num_images_to_process = 20
total_av_pre=[]
total_av_rec=[]

# Instantiate ConfusionMatrix
confusion_matrix = ConfusionMatrix(num_classes=len(coco_labels))

# Go through all 20 images in the dataset
for i, image_path in enumerate(coco_data_path.glob('frames/image*')):
	if i >= num_images_to_process:
		break
	gt_labels, pred_labels,model= get_labels(image_path)

	confusion_matrix.process_batch(pred_labels,gt_labels)
	#confusion_matrix.plot(True,coco_labels,image_path.stem,model) #uncomment to see the confusion matrix for each image
	av_pre,av_rec=pre_rec(confusion_matrix.matrix)
	
	total_av_pre.append(av_pre)
	total_av_rec.append(av_rec)
	
average_pre = sum(total_av_pre) / len(total_av_pre)
average_rec=sum(total_av_rec) / len(total_av_pre)
print(f"The average precision of {model} is: {str(average_pre)}")
print(f"The average recall of {model} is: {str(average_rec)}")




