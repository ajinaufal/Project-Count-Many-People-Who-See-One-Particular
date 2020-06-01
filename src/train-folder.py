import os

count = 20
for x in range(49):
	os.mkdir("foto/" + str(count)) #membuat folder untuk membedakan waktu
	count += 1
