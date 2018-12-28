import cv2
import numpy as np
import matplotlib.pyplot as plt

# Gray Scale Conversion
i = cv2.imread('road1.jpg')
img_copy = np.copy(i)
gray_scale = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
cv2.imwrite('gray_scale.jpg', gray_scale)

# Noise Reduction using Gaussian Blur and then Canny
def canny(img):
	gray_scale = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
	gaussian_blur = cv2.GaussianBlur(gray_scale, (5, 5), 0)
	cv2.imwrite("gaussian_blur.jpg", gaussian_blur)
	canny = cv2.Canny(gaussian_blur, 50, 150)
	cv2.imwrite("Canny.jpg", canny)
	return canny

# Region of Interest
def roi(img):
	h = img.shape[0]
	triangle = np.array([[(0, h - 100), (500, h), (310, 175)]])
	region = np.zeros_like(img)
	cv2.fillPoly(region, triangle, 255)
	region = cv2.bitwise_and(img, region)
	return region 

def display_image(img, lanes):
	line_image = np.zeros_like(img)
	if lanes is not None:
		for line in lanes:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return line_image

def coordinates(image, line_params):
	print(line_params)
	slope , intercept = line_params
	y1 = image.shape[0]
	y2 = int(y1 * (3 / 5))
	x1 = int((y1 - intercept) / slope)
	x2 = int((y2 - intercept) / slope)
	return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
	lf = []
	rf = []
	for line in lines:
		print(line)
		x1, y1, x2, y2 = line.reshape(4)
		params = np.polyfit((x1, y1), (x2, y2), 1)
		slope = params[0]
		intercept = params[1]
		if slope < 0:
			lf.append((slope, intercept))
		else:
			rf.append((slope, intercept))
	lf_avg = np.average(lf, axis = 0)
	rf_avg = np.average(rf, axis = 0)
	lf_line = coordinates(image, lf_avg)
	rf_line = coordinates(image, rf_avg)
	return np.array([lf_line, rf_line])

canny_img = canny(img_copy)
roi_img = roi(canny_img)
cv2.imwrite("Roi.jpg", roi_img)
lanes = cv2.HoughLinesP(roi_img, 2, np.pi / 180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
# averaged_lines = average_slope_intercept(img_copy, lanes)
line_img = display_image(img_copy, lanes)
combined = cv2.addWeighted(img_copy, 0.8, line_img, 1, 1)
cv2.imwrite("final_output.jpg", combined)
cv2.imshow('Image', combined)
cv2.waitKey(0)
# plt.imshow(roi_img)
# plt.imshow(canny_img)
# plt.show()