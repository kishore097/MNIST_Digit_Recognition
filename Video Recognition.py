"""

Kishore Reddy and Akhil Ajikumar
CS 5330 Computer Vision
Spring 2021

This Python file includes

- Extension : Live Video Recognition - Akhil Ajikumar
"""
import numpy as np
import keras
import cv2
model = keras.models.load_model('/home/akhil/Downloads/Project5/my_model.h5')
cap = cv2.VideoCapture(0)

while(True):
	# Capture frames
	ret, frame = cap.read()

	# Grayscale frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(7,7),0,0)
	# Thresholding
	ret, threshout = cv2.threshold(blur, 113, 255, cv2.THRESH_BINARY_INV)
	openMat = cv2.morphologyEx(threshout, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(20,20)))
	contours, hierarchy = cv2.findContours(openMat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours( frame, contours, -1, (0,0,255), 3)
	centers, numbers = [], []
	for i in range(len(contours)):
		x,y,w,h = cv2.boundingRect(contours[i])
		e = w//2 if (w>h) else h//2
		x,y = x+w//2, y+h//2
		if (y-e>0 and x-e>0):
			
			img = gray[y-e:y+e, x-e:x+e] if (x-e-20<0 or y-e-20<0) else gray[y-e-20:y+e+20, x-e-20:x+e+20] 
			ret, img = cv2.threshold( img, 174, 255, cv2.THRESH_BINARY)
			
			try:
				img = cv2.resize(img, (28,28))
				centers.append([x,y])
				numbers.append(img)
			except:
				continue
	numbers = np.array(numbers)
	numbers = numbers.reshape((numbers.shape[0], 28, 28, 1))
	if len(numbers) != 0:
		res = model.predict(numbers, batch_size=len(numbers))

		for i  in range(len(res)):
			p = res[i].tolist().index(max(res[i].tolist()))
			print("predicted:", p , "at image coordinates", centers[i])
			cv2.putText(frame, str(p), (centers[i][0], centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

	# Display frame
	cv2.imshow('frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
