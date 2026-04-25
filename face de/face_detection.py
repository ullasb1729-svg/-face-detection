import cv2
#pip install opencv-python
#haarcascade_frontalface_default.xml
a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

b = cv2.VideoCapture(0) #allow me to capture video from webcam

while True:
    c_rec, d_image = b.read() #read the video frame by frame
    e_gray = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY) #convert the image to gray scale
    f = a.detectMultiScale(e_gray, 1.3, 6) #detect the faces in the image
    
    for (x1, y1, w1, h1) in f:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2) #draw a rectangle around the detected face
         
    cv2.imshow('Face Detection', d_image) #display the video with detected faces 
    h = cv2.waitKey(40) & 0xFF #wait for 40 milliseconds and check if the 'q' key is pressed
    if h == ord('q'):
        break

b.release() #release the video capture object
cv2.destroyAllWindows() #close all OpenCV windows