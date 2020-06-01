import numpy as np
import cv2
import pickle
import datetime
import os
from PIL import Image
# from time import sleep

date = datetime.datetime.now()
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(1)
count = 0

while(True):
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("trainner.yml")

	labels = {"person_name": 1}
	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}

	ret, frame = cap.read() #membaca gambar dari kamera
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #merubah menjadi grayscale
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) #membuat variable baru
	
	for (x, y, w, h) in faces:

		print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] # dengan pixel gray
		roi_color = frame[y:y+h, x:x+w] # dengan pixel berwarna

		id_, conf = recognizer.predict(roi_gray)
		if conf>=4 and conf <= 85: # recognise
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		else :
			if not os.path.exists('foto/data-capture'):
				print("Folder data-capture berhasil dibuat")
				os.makedirs('foto/data-capture')
			# sleep(3)
			img_item = "foto/data-capture/"+ str(count) + ".png" #nama file yang di save
			cv2.imwrite(img_item, roi_gray) # menyimpan gambar sesaui dengan config
			int(count)
			# int(date.day)
			# int(date.month)
			# int(date.year)
			count += 1

		color = (255, 0, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('frame',frame) #menmpilkan gambar di jendela
	if count == 1200 : # maksimal perhitungan
		break
	elif cv2.waitKey(20) & 0xFF == ord('q'):
		break

#fungsi exit untuk keluar dari program
cap.release()
cv2.destroyAllWindows()