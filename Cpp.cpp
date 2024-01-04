#pragma execution_character_set("utf-8")
// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <iostream>
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif
#include <list>
#include <opencv2/core/utils/logger.hpp>

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;


// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1.00;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);


// Draw the predicted bounding box.
void draw_label(Mat& input_image, string label, int left, int top)
{
	// Display the label at the top of the bounding box.
	int baseLine;
	Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
	top = max(top, label_size.height);
	// Top left corner.
	Point tlc = Point(left, top);
	// Bottom right corner.
	Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
	// Draw black rectangle.
	rectangle(input_image, tlc, brc, BLACK, FILLED);
	// Put the label on the black rectangle.
	putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}


vector<Mat> pre_process(Mat &input_image, Net &net)
{
	// Convert to blob.
	Mat blob;
	blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

	net.setInput(blob);

	// Forward propagate.
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	return outputs;
}


Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name) 
{
	
	
	// Initialize vectors to hold respective outputs while unwrapping detections.
	vector<int> class_ids;
	vector<float> confidences;
	vector<Rect> boxes; 

	// Resizing factor.
	float x_factor = input_image.cols / INPUT_WIDTH;
	float y_factor = input_image.rows / INPUT_HEIGHT;

	float *data = (float *)outputs[0].data;

	const int dimensions = 85;
	const int rows = 25200;
	// Iterate through 25200 detections.
	for (int i = 0; i < rows; ++i) 
	{
		float confidence = data[4];
		// Discard bad detections and continue.
		if (confidence >= CONFIDENCE_THRESHOLD) 
		{
			float * classes_scores = data + 5;
			// Create a 1x85 Mat and store class scores of 80 classes.
			Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
			// Perform minMaxLoc and acquire index of best class score.
			Point class_id;
			double max_class_score;
			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			// Continue if the class score is above the threshold.
			if (max_class_score > SCORE_THRESHOLD) 
			{
				// Store class ID and confidence in the pre-defined respective vectors.

				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				// Center.
				float cx = data[0];
				float cy = data[1];
				// Box dimension.
				float w = data[2];
				float h = data[3];
				// Bounding box coordinates.
				int left = int((cx - 0.5 * w) * x_factor);
				int top = int((cy - 0.5 * h) * y_factor);
				int width = int(w * x_factor);
				int height = int(h * y_factor);
				// Store good detections in the boxes vector.
				boxes.push_back(Rect(left, top, width, height));
			}

		}
		// Jump to the next column.
		data += 85;
	}

	// Perform Non Maximum Suppression and draw predictions.
	vector<int> indices;
	NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
	for (int i = 0; i < indices.size(); i++) 
	{
		int idx = indices[i];
		Rect box = boxes[idx];

		int left = box.x;
		int top = box.y;
		int width = box.width;
		int height = box.height;
		// Draw bounding box.
		rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

		// Get the label for the class name and its confidence.
		string label = format("%.2f", confidences[idx]);
		label = class_name[class_ids[idx]] + ":" + label;
		// Draw class labels.
		draw_label(input_image, label, left, top);
	}
	return input_image;
	

}

void saveImageList(const vector<string>& imagePaths) {
    ofstream outFile("image_list.txt");
    if (outFile.is_open()) {
        for (const auto& path : imagePaths) {
            outFile << path << endl;
        }
        outFile.close();
    }
}


int main()
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	//create a image list
	list<Mat> imageList;

	// Load class list.
	vector<string> class_list;
	ifstream ifs("C:/medtro/coco.names");
	string line;

	while (getline(ifs, line))
	{
		class_list.push_back(line);
	}

	//C++ code does not give empty directory error for class_list. This if statement makes sure class_list is found. 
	if ( class_list.size() == 0){
		cout << "The class list directory is not correct. Please make sure to specify the full directory. "  << endl;
		return EXIT_FAILURE;
	}
	
	// // Load model. OpenCV gives error when it can not read the onnx file so no if caluse is needed. 
	Net net;
	net = readNet("C:/medtro/models/yolov5s.onnx"); 

	
	string folderPath = "C:/medtro/data/frames";
	vector<string> imagePaths;

	#ifdef _WIN32
		WIN32_FIND_DATA findFileData;
		HANDLE hFind = FindFirstFile((folderPath + "/*.*").c_str(), &findFileData);

		if (hFind != INVALID_HANDLE_VALUE) {
			do {
				// Check if the file is a regular file and an image
				if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) &&
					(strstr(findFileData.cFileName, ".jpg") || strstr(findFileData.cFileName, ".png"))) {
					// Read the image using cv::imread
					cv::Mat image = cv::imread(folderPath + "/" + findFileData.cFileName);

					if (!image.empty()) {
						
						vector<Mat> detections;
						detections = pre_process(image, net);
						
						Mat img = post_process(image.clone(), detections, class_list);

						vector<double> layersTimes;
						double freq = getTickFrequency() / 1000;
						double t = net.getPerfProfile(layersTimes) / freq;
						string label = format("Inference time : %.2f ms", t );
						string label2 = format("C++ Result" );
						Size targetSize(INPUT_HEIGHT, INPUT_WIDTH);
						Mat resizedImg;
                		resize(img, resizedImg, targetSize);
						putText(resizedImg, label, Point(20, 50), FONT_FACE, FONT_SCALE, RED,2);
						putText(resizedImg, label2, Point(20, 80), FONT_FACE, FONT_SCALE, RED,2);
						imageList.push_back(resizedImg);
						// imshow("Output_g++", resizedImg);
						// waitKey(0);
						destroyAllWindows();
						
					} else {
						std::cerr << "Error reading image: " << findFileData.cFileName << std::endl;
					}
				}
			} while (FindNextFile(hFind, &findFileData) != 0);

			FindClose(hFind);
		} else {
			std::cerr << "The folder " << folderPath << " does not exist." << std::endl;
		}
	#else
		DIR* dir = opendir(folderPath.c_str());

		if (dir != nullptr) {
			struct dirent* entry;

			// Loop through each file in the folder
			while ((entry = readdir(dir)) != nullptr) {
				// Check if the file is a regular file and an image
				if (entry->d_type == DT_REG &&
					(strstr(entry->d_name, ".jpg") || strstr(entry->d_name, ".png"))) {
					// Read the image using cv::imread
					cv::Mat image = cv::imread(folderPath + "/" + entry->d_name);

					if (!image.empty()) {
						vector<Mat> detections;
						detections = pre_process(image, net);
						
						Mat img = post_process(image.clone(), detections, class_list);

						vector<double> layersTimes;
						double freq = getTickFrequency() / 1000;
						double t = net.getPerfProfile(layersTimes) / freq;
						string label = format("Inference time : %.2f ms", t );
						string label2 = format("C++ Result" );
						Size targetSize(INPUT_HEIGHT, INPUT_WIDTH);
						Mat resizedImg;
                		resize(img, resizedImg, targetSize);
						putText(resizedImg, label, Point(20, 50), FONT_FACE, FONT_SCALE, RED,2);
						putText(resizedImg, label2, Point(20, 80), FONT_FACE, FONT_SCALE, RED,2);
						imageList.push_back(resizedImg);
					} else {
						std::cerr << "Error reading image: " << entry->d_name << std::endl;
					}
				}
			}

			closedir(dir);
		} else {
			std::cerr << "The folder " << folderPath << " does not exist." << std::endl;
		}
	#endif
	//cout << "Number of images in the list: " << imageList.size() << endl;

	//turn images into binary so they cna be read from the python file
	 
    ofstream outFile("C:/medtro/c_results.bin", ios::binary);
    if (outFile.is_open()) {
        for (const auto& image : imageList) {
            // Write the image size
            int rows = image.rows;
            int cols = image.cols;
            outFile.write(reinterpret_cast<char*>(&rows), sizeof(int));
            outFile.write(reinterpret_cast<char*>(&cols), sizeof(int));

            // Write the image data
            outFile.write(reinterpret_cast<char*>(image.data), image.total() * image.elemSize());
        }
        outFile.close();
    } else {
        cerr << "Error opening file for writing." << endl;
        return 1;
    }
	return 0; 
}