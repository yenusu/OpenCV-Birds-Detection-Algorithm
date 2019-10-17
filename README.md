#### Steps To Follow while understanding the Algorithm #####

Birds Recorgnition algorithm (Python)

import open cv

capture video from live camera

load cascade classfier (XML file)

while(TRUE)
	capture video frame
	convert frame to gray scale
	detect birds in gray scale image
	count detected birds
	print detected birds

Draw a green triangle around a detected bird
Display the results of the frame

Release the frame capture
