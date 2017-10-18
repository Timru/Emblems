import FeatureExtraction as fe
import cv2
import copy

# Detect with Sift
def detectEmblemSIFT(imagePath, precision):
	image = cv2.imread(imagePath)
	templates = fe.createTemplateListFeatureExtraction(False)
	templatesCopy = copy.deepcopy(templates)

	fe.detectKeyPointsWithSIFT(image, templatesCopy, True, precision)
	for template in templatesCopy:
		imageCopy = image.copy()
		template.calculateConfidence(templatesCopy, 0.00)
		
		for position in template.positions:
			# print a rectangle for each found position
			cv2.rectangle(imageCopy, (position[0]-5,position[1]-5), (position[0]+5,position[1]+5), (0, 0, 255), 2)
			
		# Show image
		cv2.imshow(template.name,imageCopy)
		cv2.waitKey()
		print template.name, 'emblem is found with confidence of:', template.confidence, len(template.positions)

# Detect with template matching
def detectEmblemTMPM():
	print 'to do'
	
# Update training data
# Call this method after new training images have been inserted!
def updateTrainingData():
	fe.createTemplateConfiguration('/data/training/')
	templates = fe.createTemplateListFeatureExtraction(True)
	
# to do
if __name__ == '__main__':
	
	
	updateTrainingData()
	detectEmblemSIFT('Data/testing/evaluationSheet.png', 0.6)