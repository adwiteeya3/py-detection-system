import cv2

#Image file
img_file= "images\cars8.jpg"
#video= cv2.VideoCapture('images\car_video.mp4')
video= cv2.VideoCapture('images\ped_video.mp4')

#pre-trained car and pedestrian classifier
car_tracker_file= 'car_classifier.xml'
pedestrian_tracker_file= 'ped_classifier.xml'

#create car and pedestrian classifier
car_tracker= cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker= cv2.CascadeClassifier(pedestrian_tracker_file)

#run forever until car stops
while True:
    #to read the current frame
    (read_successful, frame)= video.read()

    if read_successful:
        #Must convert to grayscale
        grayscaled_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrians
    cars= car_tracker.detectMultiScale( grayscaled_frame)
    pedestrians= pedestrian_tracker.detectMultiScale( grayscaled_frame)

    #drawing rectangles and squares around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
    
    #drawing rectangles and squares around the pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 2)    
    
    #display the image with the faces spotted in black and white
    cv2.imshow('ped_car_tracking', frame)

    #display for some time
    key= cv2.waitKey(1)

    #stop if Q or q key is pressed
    if key==81 or key==113:
        break

#Release the videocapture object
video.release()



'''
#create opencv image
img= cv2.imread(img_file)

#converting image to grayscale (needed for haar cascade) | BGR=RGB backwards
black_n_white= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#create car classifier
car_tracker= cv2.CascadeClassifier(classifier_file)

#detect cars
cars= car_tracker.detectMultiScale(black_n_white)

#drawing rectangles and squares around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)

#print(cars)


#display the image with the faces spotted in black and white
cv2.imshow('ped_car_tracking', img)

#display for some time
cv2.waitKey()

'''

#print('so far so good!')