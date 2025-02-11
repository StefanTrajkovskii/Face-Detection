import cv2

# Replace 'image.jpg' with the path to your image file
image = cv2.imread('yuki.jpg')

if image is None:
    print("Error: Could not find or open 'yuki.jpg'.")
else:
    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
