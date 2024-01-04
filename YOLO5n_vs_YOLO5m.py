import cv2
import numpy as np
from M import YOLOm_results
from N import YOLOn_results
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


#display the images side by side
def display_images_side_by_side(p_images, c_images):
	# Check if the number of images in both lists is the same
	if len(p_images) != len(c_images):
		print("Error: Number of images in the two lists must be the same.")
		return

	# Display the images side by side
	for i in range(len(p_images)):
		img1 = p_images[i]
		img2 = c_images[i]

		# Resize images if needed to ensure they have the same height
		if img1.shape[0] != img2.shape[0]:
			min_height = min(img1.shape[0], img2.shape[0])
			img1 = img1[:min_height, :]
			img2 = img2[:min_height, :]

		# Concatenate images horizontally
		concatenated_img = np.concatenate((img1, img2), axis=1)

		# Display the concatenated image
		cv2.imshow(f"Image Pair {i+1}", concatenated_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

#display the average time
def display_time(n_time,m_time):
	average_n = sum(n_time) / len(n_time)
	average_m= sum(m_time) / len(m_time)

	data = {'YOLOv5 Model': ['Nano', 'Medium'],
			'Average Time': [average_n, average_m]}

	df = pd.DataFrame(data)

	# Plotting with seaborn
	sns.barplot(x='YOLOv5 Model', y='Average Time', data=df)

	# Adding labels
	plt.xlabel('YOLOv5 Model')
	plt.ylabel('Average Time (ms)')
	plt.title('Average Time Comparison Between YOLO5n and YOLO5m')

	# Show the plot
	plt.show() 

#main function
if __name__ == "__main__":
	frameFolder = 'data/frames'
	n_images,n_times = YOLOn_results(frameFolder)
	m_images,m_times=YOLOm_results(frameFolder)
	display_images_side_by_side(n_images, m_images)
	#display_time(n_times,m_times)
	



	
	

   





