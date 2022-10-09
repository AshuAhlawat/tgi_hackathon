import cv2

# Read the original image
ROOT = "data/"
name = 'C'
img = cv2.imread(ROOT+name+".jpg") 

img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 


# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)

# Save Image

cv2.imwrite(name+"_edge.jpg", edges)
