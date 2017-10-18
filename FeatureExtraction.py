import cv2
import numpy as np
import os.path
import Templating
import cPickle
import xml.etree.ElementTree as ET

scriptPath = os.path.realpath(__file__)
scriptPath = scriptPath[:scriptPath.find('Feature')]
scriptPath = scriptPath[:scriptPath.find('feature')]

# Erweiterung des SIFT Algorithmus
class RootSIFT:
	def __init__(self):
		# initialize the SIFT feature extractor
		self.extractor = cv2.DescriptorExtractor_create("SIFT")
 
	def compute(self, image, kps, eps=1e-7):
		# compute SIFT descriptors
		(kps, descs) = self.extractor.compute(image, kps)
 
		# if there are no keypoints or descriptors, return an empty tuple
		if len(kps) == 0:
			return ([], None)
 
		# apply the Hellinger kernel by first L1-normalizing and taking the
		# square-root
		descs /= (descs.sum(axis=1, keepdims=True) + eps)
		descs = np.sqrt(descs)
		#descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)
 
		# return a tuple of the keypoints and descriptors
		return (kps, descs)
		
def detectKeyPointsForTemplates(templates, createNewConfig, scriptpath):
	if createNewConfig:
		sift = cv2.SIFT()
		for template in templates:
			kpTemp, desTemp = sift.detectAndCompute(template.image, None)
			#print template.name, ' number of kp: ', len(kpTemp)
			rs = RootSIFT()
			kpTemp, desTemp = rs.compute(template.image, kpTemp)
			template.kpTemp = kpTemp
			#print template.name,'has', len(kpTemp)
			template.desTemp = desTemp
			np.savez(scriptpath+'/data/configs/SIFTKeypointConfig_'+template.name, desTemp)
	else:
		for template in templates:
			desTemp = []
			npzfile = np.load(scriptpath+'/data/configs/SIFTKeypointConfig_'+template.name+'.npz')

			for elem in npzfile['arr_0']:
				desTemp.append(elem)
			
			template.desTemp = np.asarray(desTemp)
			
	return templates
	
# Create list of featureExtraction templates from detectionConfig.xml
def createTemplateListFeatureExtraction(createNewConfig):
	
	#print scriptPath
	doc = ET.parse(scriptPath+'/data/configs/templates_emblem.xml')
	configTree = doc.getroot()
	
	templates = []
	templatesPath = ''
	
	for child in configTree:
		if 'featureExtraction' in child.tag:
			templatesPath = child.attrib.get('path')
			for template in child:
				templatePath = templatesPath + template.attrib.get('name') + '.jpg'
				if os.path.isfile(scriptPath+templatePath):
					image = cv2.imread(scriptPath+templatePath)
					t = Templating.Template(template.attrib.get('name'), image)
					templates.append(t)
				else:
					image = cv2.imread(templatePath)
					
					t = Templating.Template(template.attrib.get('name'), image)
					templates.append(t)

	templates = detectKeyPointsForTemplates(templates, createNewConfig, scriptPath)
	return templates
	
def detectKeyPointsWithSIFT(image, templates, useBFMatch, precision):
	img_gray = image
	sift = cv2.SIFT()
	kpImg, desImg = sift.detectAndCompute(img_gray,None)
	rs = RootSIFT()
	kpImg, desImg = rs.compute(img_gray, kpImg)

	if useBFMatch:
		#clock = timer.StopWatch()
		
		bf = cv2.BFMatcher()

		for template in templates:
			#clock.count()
			result = []
			xyCoordinates = []
			#kpTemp, desTemp = sift.detectAndCompute(template.image, None)
			matches = bf.knnMatch(desImg, template.desTemp, k=2)
			precision = 1-precision;
			for m,n in matches:
				if m.distance < precision*n.distance:
					result.append(m)
		
			for mat in result:
				point = []
				img1_idx = mat.queryIdx
				tuple = kpImg[img1_idx].pt
				point.append(int(tuple[0]))
				point.append(int(tuple[1]))

				template.positions.append(point)

	else:
		FLANN_INDEX_KDTREE = 1
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks=50) # or pass empty dictionary
		rs = RootSIFT()
		kpImg, desImg = rs.compute(img_gray, kpImg)

		for template in templates:
			result = []
			xyCoordinates = []
			flann = cv2.FlannBasedMatcher(index_params,search_params)

			#matches = bf.knnMatch(desImg, desTemp, k=2)
			matches = flann.knnMatch(desImg, template.desTemp,k=2)
			matchesMask = [[0,0] for i in xrange(len(matches))]
			
			#for m,n in matches:
			for i,(m,n) in enumerate(matches):
				if m.distance < 0.5*n.distance:
					result.append(m)
			
			for mat in result:
				point = []
				img1_idx = mat.queryIdx
				tuple = kpImg[img1_idx].pt
				point.append(int(tuple[0]))
				point.append(int(tuple[1]))

				template.positions.append(point)
			
	return templates
	
def createTemplateConfiguration(folderPath):
	root = ET.Element('root')
	child = ET.SubElement(root, "featureExtraction")
	child.set('path',folderPath)
	for fileName in os.listdir(scriptPath+folderPath):
		#print child
		fileName = fileName[:fileName.find('.')]
		templateChild = ET.SubElement(child, 'template').set('name',fileName)

	tree = ET.ElementTree(root)
	tree.write(scriptPath+'/data/configs/templates_emblem.xml')

	templates = createTemplateListFeatureExtraction(True)
	
if __name__ == '__main__':
	createTemplateConfiguration('/data/training/')
	templates = createTemplateListFeatureExtraction(True)