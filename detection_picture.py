import time
import cv2
from matplotlib import pyplot as plt

from facenet.face_contrib import *


def add_overlays(frame, faces, colors, confidence=0.4):
    if faces is not None:
        for idx, face in enumerate(faces):
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame, (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]), colors[idx], 2)
            if face.name and face.prob:
                if face.prob > confidence:
                    class_name = face.name
                else:
                    class_name = 'Unknown'
                    # class_name = face.name
                cv2.putText(frame, class_name, (face_bb[0], face_bb[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            colors[idx], thickness=2, lineType=2)
                cv2.putText(frame, '{:.02f}%'.format(face.prob * 100), (face_bb[0], face_bb[3] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], thickness=1, lineType=2)

def detection_pic(model_checkpoint, classifier):
    face_recognition = Recognition(model_checkpoint, classifier)
    for n in range(1,19):
        colors = np.random.uniform(0, 255, size=(1, 3))
        img = cv2.imread(f"./input_test_pic/{n}.jpg",cv2.IMREAD_COLOR)
        faces = face_recognition.identify(img)
        for i in range(len(colors), len(faces)):
            colors = np.append(colors, np.random.uniform(150, 255, size=(1, 3)), axis=0)
        add_overlays(img, faces, colors)
        cv2.imwrite(f'./output_test_pic/{n}.jpg',img)
        #cv2.imshow("test",img)
        # if cv2.waitKey(0) & 0xFF == ord('n'):
        #     continue
        cv2.destroyAllWindows()
if __name__ == '__main__':
    detection_pic('models', 'models/your_model.pkl')