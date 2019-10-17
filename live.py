#Author:        Kiryamwibo Yenusu
#contact:       +256 776 656 200
#Email:         kiryamwiboyenusu@gmail.com
#Education:     BSc. Softaware Engineering (Makerere University)
#Home:          Mayuge-Uganda
#github:        www.github.com/yenusu
#stackoverflow: www.stackoverflow.com/users/5442050/kiryamwibo-yenusu
#facebook:      www.facebook.com/yenusu
#twitter:       www.twitter.com/kyenusu
#Linkedin:      www.linkedin.com/yenusu
#date:          24/feb/2018

# This script will detect birds approaching a rice farm via an external camera.
# It is Tested with OpenCV3 on linux ubuntu 16.4 running Python 3 or above

import cv2
#cap = cv2.VideoCapture(0) #for automatic USBCam (0-web cam default)

class DetectBirds(object):
    def __init__(self, camera_url, mx_num_birds = 3):
        self.cap = cv2.VideoCapture(camera_url)
        self.birdsCascade = cv2.CascadeClassifier("birds1.xml")
        self.MAX_NUM_BIRDS = mx_num_birds
        self.running = True

    def detect(self):
        while self.running:
            # Capture frame-by-frame from a video
            ret, frame = self.cap.read()
            if ret:
                # convert the frame into gray scale for better analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect birds in the gray scale image
                birds = self.birdsCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.4,
                    minNeighbors=5,
                    #minSize=(10, 10),
                    maxSize=(30, 30),
                    flags = cv2.CASCADE_SCALE_IMAGE
                )
                if (len(birds)>=self.MAX_NUM_BIRDS):
                    print("Detected {0} Birds".format(len(birds)))

                # Draw a rectangle around the detected birds approaching the farm
                for (x, y, w, h) in birds:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)


                # Display the resulting frame
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
            else:
                self.running = False

        # When everything done, release the capture and go back take another one
        cap.release()
        cv2.destroyAllWindows()

# Create the haar cascade that reads the properties of objects to be detected from an already made xml file.
# The xml file is generated as a result of machine learning from all possible object samples provided.


if __name__ == "__main__":
    D = DetectBirds("birds.mp4")
    D.detect()
