import cv2
import numpy as np
from py import python_results

c_images=[]

#accesing the C results by reading the binary file
def get_C_results(binFile):
	try:
		with open(binFile, "rb") as file:
			while True:
				# Read the image size
				rows = int.from_bytes(file.read(4), byteorder='little', signed=True)
				cols = int.from_bytes(file.read(4), byteorder='little', signed=True)

				# Read the image data
				img_data = file.read(rows * cols * 3)  # Assuming 3 channels for color images
				if not img_data:
					break

				# Convert the binary data to a NumPy array and reshape it to the original image shape
				image = np.frombuffer(img_data, dtype=np.uint8).reshape((rows, cols, 3))

				c_images.append(image)

		return c_images
		

	except FileNotFoundError:
		print("File not found. Run the C++ program first.")
	except Exception as e:
		print("An error occurred while reading the binary file.", e)



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

#main function
if __name__ == "__main__":
	frameFolder = 'data/frames'
	p_images = python_results(frameFolder)
	binFile='c_results.bin'
	c_images=get_C_results(binFile)
	display_images_side_by_side(p_images, c_images)



	
	

   





