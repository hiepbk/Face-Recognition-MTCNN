import cv2
import time
# Method to generate dataset to recognize a person
def generate_dataset(img, id, img_id):
    # write image in data dir
    cv2.imwrite("./input_face/Hiep/hiep_"+str(img_id)+".jpg", img)
# Capturing real time video stream. 0 for built-in web-cams, 0 or -1 for external web-cams
video_capture = cv2.VideoCapture(0)

# Initialize img_id with 0
img_id = 0
count=0
start_time = time.time()
while (video_capture.isOpened()):
    if count == 50:
        break
    # Reading image from video stream
    _, img = video_capture.read()
    # Call method we defined above
    # Writing processed image in a new window
    end_time = time.time()
    if (end_time - start_time) > 0.5 :
        #if(cv2.waitKey(1) & 0xFF == ord('c')):
        generate_dataset(img,1,img_id)
        print("Collected ", img_id, " images")
        img_id += 1
        count += 1
        start_time = time.time()
    cv2.imshow("face detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing web-cam
video_capture.release()
# Destroying output window
cv2.destroyAllWindows()