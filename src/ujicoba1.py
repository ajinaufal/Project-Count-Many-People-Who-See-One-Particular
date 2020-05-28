import numpy as np
import cv2
import datetime
import os

date = datetime.datetime.now()
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(1)
count = 0
os.mkdir("foto/" + str(date.day) + "-" + str(date.month) + "-" + str(date.year))
while(True):
	ret, frame = cap.read() #membaca gambar dari kamera
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #merubah menjadi grayscale
	faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) #membuat variable baru
	for (x, y, w, h) in faces:
		print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] # dengan pixel gray
		roi_color = frame[y:y+h, x:x+w] # dengan pixel berwarna
		img_item = "foto/"+ str(date.day) + "-" + str(date.month) + "-" + str(date.year) + "/my-image" + str(count) + ".png" #nama file yang di save
		cv2.imwrite(img_item, roi_color) # menyimpan gambar sesaui dengan config
		int(count)
		count += 1
		color = (255, 0, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('frame',frame) #menmpilkan gambar di jendela
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

#fungsi exit untuk keluar dari program
cap.release()
cv2.destroyAllWindows()