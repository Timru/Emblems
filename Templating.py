

# Template class for storing information about template images themselves and found templates within a picture
class Template:

	def __init__(self, name, image):
		# name
		self.name = name
		
		# CV2 image reference
		self.image = image
		
		self.kpTemp = 0
		self.desTemp = 0
		
		self.confidence = 0
		
		# List of detection positions
		self.positions = []

		# Coloring scheme for highlighting (Blue, Green, Red)
		self.colors = []
		

		
	def calculateConfidence(self, templates, confidenceThreshold):
		currConfidence = 0
		finalConfidence = 0
		bestMatch = 0
		bestConfidence = 0.0
		#print self.name, len(self.desTemp), len(self.positions)
		for template in templates:
			#print template.name, self.name
			if template.name[:-1] in self.name:

				#print template.name[:-3], self.name
				currConfidence = float(len(template.positions))/len(template.desTemp)
				currConfidence = currConfidence*currConfidence
				finalConfidence = finalConfidence + currConfidence - float((finalConfidence*currConfidence))



				#print 'positive', finalConfidence
				if len(template.positions)>bestMatch:
					bestMatch = len(template.positions)

				if currConfidence > bestConfidence:
					bestConfidence = currConfidence


		self.confidence = finalConfidence
		self.bestMatch = bestMatch
		self.bestConfidence = bestConfidence