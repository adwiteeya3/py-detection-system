import cv2

#Image file
img_file= "images\car2.jpg"

#pre-trained car classifier
classifier_file= 'car_classifier.xml'

#create opencv image
img= cv2.imread(img_file)

#create car classifier
car_tracker= cv2.CascadeClassifier(classifier_file)



#display the image with the faces spotted
cv2.imshow('ped_car_tracking', img)

#display for some time
cv2.waitKey()

print('so far so good!')