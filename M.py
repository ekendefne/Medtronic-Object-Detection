import cv2
import numpy as np
import os

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45

# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


def draw_label(input_image, label, left, top):
	"""Draw text onto image at location."""
	
	# Get text size.
	text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
	dim, baseline = text_size[0], text_size[1]
	# Use text size to create a BLACK rectangle. 
	cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
	# Display text inside the rectangle.
	cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)


def pre_process(input_image, net):
	# Create a 4D blob from a frame.
	blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

	# Sets the input to the network.
	net.setInput(blob)

	# Runs the forward pass to get output of the output layers.
	output_layers = net.getUnconnectedOutLayersNames()
	outputs = net.forward(output_layers)
	# print(outputs[0].shape)

	return outputs


def post_process(input_image, outputs):
	classesFile = "coco.names"
	classes = None
	with open(classesFile, 'rt') as f:
		classes = f.read().rstrip('\n').split('\n')
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

		# Discard bad detections and continue.
		if confidence >= CONFIDENCE_THRESHOLD:
			classes_scores = row[5:]

			# Get the index of max class score.
			class_id = np.argmax(classes_scores)

			#  Continue if the class score is above threshold.
			if (classes_scores[class_id] > SCORE_THRESHOLD):
				confidences.append(confidence)
				class_ids.append(class_id)

				cx, cy, w, h = row[0], row[1], row[2], row[3]

				left = int((cx - w/2) * x_factor)
				top = int((cy - h/2) * y_factor)
				width = int(w * x_factor)
				height = int(h * y_factor)
			  
				box = np.array([left, top, width, height])
				boxes.append(box)

	# Perform non maximum suppression to eliminate redundant overlapping boxes with
	# lower confidences.
	indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
	for i in indices:
		box = boxes[i]
		left = box[0]
		top = box[1]
		width = box[2]
		height = box[3]
		cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
		label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
		draw_label(input_image, label, left, top)

	return input_image


def YOLOm_results(frameFolder):
	YOLOm=[]
	m_time=[]
	# Give the weight files to the model and load the network using them.
	modelWeights = "models/yolov5m.onnx"
	net = cv2.dnn.readNet(modelWeights)
	
	# Check if the folder exists
	if os.path.exists(frameFolder) and os.path.isdir(frameFolder):
		# List all files in the folder
		image_files = [f for f in os.listdir(frameFolder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

		# Loop through each image file and read it using cv2.imread
		for image_file in image_files:
			file_path = os.path.join(frameFolder, image_file)
			img = cv2.imread(file_path)

			if img is not None:
				detections = pre_process(img, net)
				img = post_process(img.copy(), detections)

				# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
				t, _ = net.getPerfProfile()
				m_time.append(t)
				label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
				label2='YOLO5m Result'
				#img = cv2.resize(img, (INPUT_HEIGHT,INPUT_WIDTH))
				cv2.putText(img, label, (20, 50), FONT_FACE, FONT_SCALE, RED, 2, cv2.LINE_AA)
				cv2.putText(img, label2, (20, 80), FONT_FACE, FONT_SCALE, RED, 2, cv2.LINE_AA)
				YOLOm.append(img)
				#print(img.shape[0],img.shape[1])
				# cv2.imshow(f"Image Pair", img)
				# cv2.waitKey(0)
				# cv2.destroyAllWindows()		
			else:
				print(f"Error reading image: {image_file}")
	else:
		print(f"Please make sure the directory of frames is correct in 'Py.py' code.")

	return YOLOm,m_time


if __name__ == '__main__':
	frameFolder = 'data/frames'
	YOLOm,m_time=YOLOm_results(frameFolder)
	print(f"Done compiling")

	