# importing the module
import cv2
import rasterio
import numpy as np


def click_event(event, x, y, flags, params): 
	'''

	'''
	# checking for left mouse clicks.
	if event == cv2.EVENT_LBUTTONDOWN:
		# displaying the coordinates.
		# on the Shell.
		print(x, ' ', y) 
	# checking for right mouse clicks.
	if event==cv2.EVENT_RBUTTONDOWN:
		# displaying the coordinates.
		# on the Shell 
		print(x, ' ', y) 

		# displaying the coordinates 
		# on the image window 
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0] 
		g = img[y, x, 1] 
		r = img[y, x, 2] 
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)


# driver function 
if __name__ == "__main__":
	
	# path of the image
	path_image = 'data/raw/crop_SENTINEL2B_20211109-110724-649_L2A_T31UDQ_C_V3-0.tif'
	# reading the image	
	src = rasterio.open(path_image)
	red = src.read(2)
	green = src.read(3)
	blue = src.read(4)
	nri = src.read(1)
	
	red = ((red-np.min(red))/(np.max(red)-np.min(red)))*510
	green = ((green-np.min(green))/(np.max(green)-np.min(green)))*510
	blue = ((blue-np.min(blue))/(np.max(blue)-np.min(blue)))*510
	img = np.dstack((blue, green, red))
	# displaying the image
	cv2.imshow('image', img.astype(np.uint8))

	# setting mouse handler for the image
	# and calling the click_event() function
	cv2.setMouseCallback('image', click_event)

	# wait for a key to be pressed to exit
	cv2.waitKey(0)

	# close the window
	cv2.destroyAllWindows()
