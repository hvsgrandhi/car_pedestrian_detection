import cv2

video = cv2.VideoCapture('video2.mp4')

#pretrained car and pedestrian classifier
car_tracker_file = 'car_detection.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'

#creation of car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#run forever until video stops
while True:

    #reading a frame from the video
    (read_successfull, frame) = video.read()

    #safe coding
    if read_successfull:
        #converting each frame into black and white(grayscale)
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #cv2.rectangle(image, (x n y corrdinates), (x+w n y+h coordinate), (color of the rectangle in form of bgr), (thickness of the rectangle))

    #draw rectangles around the pedestrians
    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    #displays the images with cars
    cv2.imshow('The name of the window', frame)

    #do not auto close the window(closses on any keypress)
    key = cv2.waitKey(1)

    # ascii code for Q to exit the video
    if key == 81 or key == 113:
        break

#done reading the file, release all the resources, release the videoCapture Object
video.release()