import cv2
import numpy as np
import csv
from matplotlib import pyplot as plt



#dst = cv2.cornerHarris(gray,2,3,0.04)

def getTrafficCongestionLevel(img,mask,height,width):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	
	coners = cv2.goodFeaturesToTrack(gray,300,0.01,5)  #50
	dst = np.zeros((height,width), np.uint8)
	X,Y = np.mgrid[:height, :width]
	for e in coners:
		r = int(e[0,0]);
		c = int(e[0,1]);
		#print r,c
		if mask[c,r]==255:
			#img[c,r] = [0,0,255]
			cv2.circle(img, (r,c), 4, (255, 0, 0), 1)
			dst = dst + (((c-X)**2 + (r-Y)**2)<100);
	
	dst[dst>0] = 255
	congestionLevel = (float(sum(sum(dst>0)))/float(sum(sum(mask>0))))*100;
	print 'Congestion Level: ',congestionLevel;
	cv2.imshow('dsts',img);
	return congestionLevel,dst;
			

	# img = cv2.dilate(img,None)
	#result is dilated for marking the corners, not important
	#dst = cv2.dilate(dst,None)
	#img[dst] = [0,0,255]
	# Threshold for an optimal value, it may vary depending on the image.
	#img[dst>0.1*dst.max()]=[0,0,255]

	cv2.imshow('dst',img)
	


def getROI(filename,height,width):
	#print height
	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		mask = np.zeros((height,width), np.uint8)
		#print mask.shape
		i =0;
		j=0;
		for row in reader:
			
			for column in row:
					#print column
					if (int(column) == 1):
						mask[i,j] = 255;
						#print i,j
					j=j+1;
			j=0
			i = i+1;
	return mask

	
		
		
def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
		
        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
		
		

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out

def DetectCongestion(mask):
	img1 =cv2.imread('wood11.jpg');
	img2 =cv2.imread('wood12.jpg');
	
	img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
	#img1 = np.float32(img1)
	
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	#img2 = np.float32(img2)
	
	surf = cv2.SURF(1000)
	
	kp1, des1 = surf.detectAndCompute(img1,mask)
	kp2, des2 = surf.detectAndCompute(img2,mask)
	
	bf = cv2.BFMatcher()
	
	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	
	#print matches[5].imgIdx
	#img3 = drawMatches(img1,kp1,img2,kp2,matches)  #10
	getEligibleMatchedPoints(img1,kp1,img2,kp2,matches[:10],mask)    #10
	#plt.imshow(img3),plt.show()
	
	
def getEligibleMatchedPoints(img1,kp1,img2,kp2,matches,mask):

	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]
	
	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
	
	out[:rows1,:cols1] = np.dstack([img1, img1, img1])
	
	out[:rows2,cols1:] = np.dstack([img2, img2, img2])
	
	
	threshold = 2000;
	eligibleMatches1 = [];
	eligibleMatches2 = [];
	for mat in matches:
		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx
		(x1,y1) = kp1[img1_idx].pt;
		(x2,y2) = kp2[img2_idx].pt;
		
		gradient = (y1-y2)/(x1-x2);
		
		
		
		 
		if (gradient<=0 and mask[y1,x1]==255 and mask[y2,x2]==255):
			isDistinct = 1;
			for e in eligibleMatches1:
				if ((x1-e[0])**2 + (y1-e[1])**2)<threshold:
					isDistinct = 0;
		
			if (isDistinct == 1):		
				eligibleMatches1.append((x1,y1))
				eligibleMatches2.append((x2,y2))
				#print x1,y1
				cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
				cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
				
				cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
	
	print 'No. of Same Vehicles Detected: ',len(eligibleMatches2)
	cv2.imshow('Matched Features 2', out)
	return eligibleMatches1,eligibleMatches2
    
	
		
#Program Starts Here.........
filename = 'wood11.jpg'
img = cv2.imread(filename)
#detectCorners(img)
dim = img.shape;

mask  = getROI('BW2.csv',dim[0],dim[1])
congestionLevel,dst=getTrafficCongestionLevel(img,mask,dim[0],dim[1])
kernel = np.ones((19,19),np.uint8)
dst = cv2.dilate(dst,kernel,iterations = 1)
cv2.imshow('imagw',dst);

DetectCongestion(mask)   #mask


if cv2.waitKey(0) & 0xff == 27:
		cv2.destroyAllWindows()