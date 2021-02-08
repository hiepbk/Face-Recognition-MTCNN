import tensorflow as tf
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn import MTCNN
from imutils import paths
from matplotlib import pyplot as plt
import cv2
import glob
model = tf.keras.models.load_model("your_model.model")
IMG_SIZE = 100

def detect_face(filename, required_size=(IMG_SIZE, IMG_SIZE)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    print(x1,y1,width,height)
    print(results[0]['box'])
    x2, y2 = x1 + width + width // 6, y1 + height + height // 6
    print(width // 6)

    if y1 > height // 6:
        y1_new = y1 - height // 6
    else:
        y1_new = y1
    if x1 > width // 6:
        x1_new = x1 - width // 6
    else:
        x1_new = x1
    print(x1_new,y1_new)

    # extract the face
    face = pixels[y1_new:y2, x1_new:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    # face_array = cv2.cvtColor(face_array, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(face_array, (IMG_SIZE, IMG_SIZE))
    new_array = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    prediction = model.predict([new_array])
    prediction = list(prediction[0])
    print(prediction)
    # get dimensions of image
    dimensions = pixels.shape
    height_IN = pixels.shape[0]
    width_IN = pixels.shape[1]
    print(x1_new,y1_new,width_IN,height_IN,width,height)
    cv2.putText(pixels,'{:.02f}%'.format(max(prediction) * 100),(x1_new ,y2 +width//3 ),cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color=(255,0,0), thickness=2, lineType=2)

    pixels = cv2.rectangle(pixels, (x1_new, y1_new),
                           (x1 + width + width // 6, y1 + height + height // 6),
                           (255, 0, 0), 2)
    plt.imshow(pixels)
    plt.show()
    print(prediction[prediction.index(max(prediction))] * 100, '%')
    # return new_array
for img in glob.glob("2.FDDB/originalPics/2003/01/13/big/*.jpg"):
    detect_face(img)



