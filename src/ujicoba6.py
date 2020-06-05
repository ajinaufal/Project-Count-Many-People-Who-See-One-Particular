import numpy as np
import cv2
import pickle
import datetime
import os
from PIL import Image
from time import sleep

date = datetime.datetime.now()

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
cap = cv2.imread('train7.jpg') # 0 adalah kamera internal dan 1 adalah kamera eksternal
gray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY) #merubah menjadi grayscale
faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) #membuat variable baru

recognizer = cv2.face.LBPHFaceRecognizer_create()# menggunakan LBPH
count = 0

# people = 0

# def learningulang():# loop learning
# 	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 	if not os.path.exists('dataset'): #pembuatan foldet dataset
# 		print("Folder dataset berhasil dibuat")
# 		os.makedirs('dataset') # pembuatan folder dataset
# 	image_dir = os.path.join(BASE_DIR, "dataset")

# 	face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
# 	recognizer = cv2.face.LBPHFaceRecognizer_create()

# 	current_id = 0
# 	label_ids = {}
# 	y_labels = []
# 	x_train = []

# 	for root, dirs, files in os.walk(image_dir):
# 		for file in files:
# 			if file.endswith("png") or file.endswith("jpg"):
# 				path = os.path.join(root, file)
# 				label = os.path.basename(root).replace(" ","-").lower()
# 				#print(label, path)
# 				if not label in label_ids:
# 					label_ids[label] = current_id
# 					current_id +=1
# 				id_ = label_ids[label]
# 				#print(label_ids)
# 				#y_labels.append(label)#some number
# 				#x_train.append(path) #verify this imafe, turn into a Numpy array, Gray
# 				pil_image = Image.open(path).convert("L") #grayscale
# 				size = (550, 550)
# 				final_image = pil_image.resize(size, Image.ANTIALIAS)
# 				image_array = np.array(final_image, "uint8")
# 				#print(image_array)
# 				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5) #deteksi wajah

# 				for (x,y,w,h) in faces:
# 					roi = image_array[y:y+h, x:x+w]
# 					x_train.append(roi)
# 					y_labels.append(id_)


# 	#print(y_labels)
# 	#print(x_train)
# 	with open("labels.pickle", 'wb') as f:
# 		pickle.dump(label_ids, f)
	
# 	recognizer.train(x_train, np.array(y_labels))
# 	recognizer.save("trainner.yml")# menyimpan hasil training

# learningulang()
# while(True):

if len(faces) == 0:
    print ("No faces found")

else:
	labels = {"person_name": 1}
	with open("labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}
	#ret, cap = cap.read() #membaca gambar dari kamera

	
	recognizer.read("trainner.yml")# membaca hasil training
	for (x, y, w, h) in faces:
		#print(x,y,w,h)
		roi_gray = gray[y:y+h, x:x+w] # dengan pixel gray
		roi_color = cap[y:y+h, x:x+w] # dengan pixel berwarna
		id_, conf = recognizer.predict(roi_gray)
		if conf>=4 and conf <= 85: # recognise
			#print(id_)
			#print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (255, 255, 255)
			stroke = 2
			cv2.putText(cap, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			print("recognize berhasil" + labels[id_])
		else :
			if not os.path.exists('dataset'):
				print("Folder data-capture berhasil dibuat")
				os.makedirs('dataset')
			if not os.path.exists('dataset/' + str(count)):
				print("Folder dataset/xx berhasil dibuat")
				os.makedirs('dataset/'+ str(count))
			img_item = "dataset/"+ str(count) + "/" + "people-" + str(count)  + ".png" #nama file yang di save
			cv2.imwrite(img_item, roi_color) # menyimpan gambar sesaui dengan config
			int(count)
			count += 1
			print ("recognize gagal dan berhasil mengambil gambar")
			# learningulang()
			sleep(3)
			# int(people)
			# int(date.day)
			# int(date.month)
			# int(date.year)
		color = (255, 0, 0) #BGR 0-255
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(cap, (x,y), (end_cord_x, end_cord_y), color, stroke)
	cv2.imshow('cap',cap) #menmpilkan gambar di jendela
	print (count)
	# if count == 100 : # maksimal perhitungan
	# 	break
	# elif cv2.waitKey(20) & 0xFF == ord('q'):
	# 	break


	#fungsi exit untuk keluar dari program
	# cap.release()
	cv2.waitKey(0)
	cv2.destroyAllWindows()