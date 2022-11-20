import imutils
from imutils import paths
from imutils.video import VideoStream
from imutils.video import FPS
from gpiozero import AngularServo
from pushbullet import Pushbullet
from time import sleep
import face_recognition
import pickle
import time
import cv2
import os
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images('dataset'))
knownEncodings = []
knownNames = []
servo = AngularServo(18, min_pulse_width=0.0006, max_pulse_width=0.0023)
for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	boxes = face_recognition.face_locations(rgb,model='hog')
	encodings = face_recognition.face_encodings(rgb, boxes)
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open('encodings.pickle', "wb")
f.write(pickle.dumps(data))
f.close()
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open('encodings.pickle', "rb").read())
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
print("[INFO] starting video stream...")
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
fps = FPS().start()
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
	boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
	encodings = face_recognition.face_encodings(rgb, boxes)
	names = []
	for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],encoding)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
	for encoding in encodings:
		matches = face_recognition.compare_faces(data["encodings"],encoding)
		name = "Unknown"
		if True in matches:
		    servo.angle = 90
		    sleep(2)
		    servo.angle = 0
		    sleep(2)
	for encoding in encodings:
	    matches = face_recognition.compare_faces(data["encodings"],encoding)
	    name = "Unknown"
	    if False in matches:
		    cv2.imwrite(filename='saved_img.jpg', img=frame)
		    webcam.release()
		    img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
		    img_new = cv2.imshow("Captured Image", img_new)
		    cv2.waitKey(1650)
		    cv2.destroyAllWindows()
		    print("Processing image...")
		    img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
		    print("Converting RGB image to grayscale...")
		    img_resized = cv2.imwrite(filename='Thread_people.jpg', img=img_)
		    print("Image saved!")
	for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"],encoding)
            name = "Unknown"
            if False in matches:
                API_KEY = "o.tqaliMNTRqQ2a5kTNIaqB58g1JHMyI4m"
                file = "thread.txt"
                with open(file,mode = 'r') as f:
                    text = f.read()
                pb = Pushbullet(API_KEY)
                push = pb.push_note("Important message",text)
	for ((top, right, bottom, left), name) in zip(boxes, names):
		cv2.rectangle(frame, (left, top), (right, bottom),
			(0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15
		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
			0.75, (0, 255, 0), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
cv2.destroyAllWindows()
vs.stop()
