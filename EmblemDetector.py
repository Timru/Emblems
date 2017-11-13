import FeatureExtraction as fe
import cv2
import copy
import numpy as np

# Detect with Sift
def detectEmblemSIFT(imagePath, precision, updateTrainingData):
	image = cv2.imread(imagePath)
	if updateTrainingData:
		updateTrainingData()
	templates = fe.createTemplateListFeatureExtraction(False)
	templatesCopy = copy.deepcopy(templates)
	templatesCopy = sorted(templatesCopy, key=lambda template: template.positions, reverse=True)
	fe.detectKeyPointsWithSIFT(image, templatesCopy, True, precision)
	for template in templatesCopy:
		imageCopy = image.copy()
		template.calculateConfidence(templatesCopy, 0.00)
		
		#for position in template.positions:
			# print a rectangle for each found position
		#	cv2.rectangle(imageCopy, (position[0]-5,position[1]-5), (position[0]+5,position[1]+5), (0, 0, 255), 2)
			
		# Show image
		#cv2.imshow(template.name,imageCopy)
		#cv2.waitKey()
	templatesCopy = sorted(templatesCopy, key=lambda template: len(template.positions), reverse=True)
	for template in templatesCopy:
		print template.name, 'emblem is found with confidence of:', len(template.positions)

# Detect with template matching
def detectEmblemTMPM():
	img = cv2.imread('data/training/b3.jpg',0)
	img2 = img.copy()
	template = cv2.imread('data/training/b3.jpg',0)
	w, h = template.shape[::-1]
	res = cv2.matchTemplate(img2, template,cv2.TM_SQDIFF_NORMED)
	print res
	# Filter location points by tolerance value
	loc = np.where( res <= 0.05)
	zipped = zip(*loc[::-1])
	print len(zipped)
	
	
# Update training data
# Call this method after new training images have been inserted!
def updateTrainingData():
	fe.createTemplateConfiguration('/data/training/')
	templates = fe.createTemplateListFeatureExtraction(True)
	

# Main method
if __name__ == '__main__':
	
	# Compares an input image against the training images
	# First param: input image path
	# Second param: tolerance/precision
	# Third param: if new images have been inserted in the training folder set it to true, after running it once set it to false
	detectEmblemSIFT('Data/testing/b1.jpg', 0.2, False)
