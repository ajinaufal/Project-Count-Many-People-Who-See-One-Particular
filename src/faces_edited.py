import numpy as np 
import cv2
import pickle
import datetime


#time_ = datetime.datetime.now()
#print(x.strftime("%x"))

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(1)

while(True):
	i = 0
	# capture frame-by-frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5) #deteksi wajah
	for (x, y, w, h) in faces:
		i += 1
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w] #(ycord_start, ycord_end) croping foto
		
		id_, conf = recognizer.predict(roi_color)
		if conf>=4: and conf <= 85: # recognise
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


		img_item = "my-image.png"
		#cv2.imwrite(img_item, roi_color)
		#cv2.imwrite(x.strftime("%x") + "/people" + str(i) + ".png", roi_color)
		color = (255, 0, 0) #bgr 0-255
		stroke = 2
		end_cord_x = x+w
		end_cord_y = y+h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	#display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#when everityng done, realese the capture
cap.realese()
cv2.destroyAllWindows()