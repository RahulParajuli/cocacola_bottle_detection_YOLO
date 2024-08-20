
import pickle
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
from PIL import Image

YOLO_PREDICTION_LABEL_PATH = "runs/detect/predict/labels/"
YOLO_MODEL = YOLO("channel/yolov8n.pt")
CNN_MODEL = pickle.load(open('modelRunner/cnn_model.pkl', 'rb'))

def responseHelper(code, message, data):
    return {
        "code": code,
        "message": message,
        "data": data
    }

class DetectandClassifyCoke():
    
    def __init__(self) -> None:
        self.target_height = 256
        self.target_width = 256
        self.class_names = ['Coke', 'Nocoke']
    
    def get_bottle_detection_with_yolo(self):

        image = np.array(Image.open(self.image))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite("temp_image.jpg", image)
        h, w = image.shape[0], image.shape[1]
        YOLO_MODEL.predict("temp_image.jpg", save = True, save_txt = True)
        cropped_images = self.crop_bottle_detection(image_name="temp_image.jpg", image = image, width = w, height = h)
        return cropped_images

        
    
    def get_bottle_classification_with_cnn(self, image = None):

        final_prediction = []
        img = tf.keras.utils.load_img(
            image[1], target_size=(self.target_height, self.target_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = CNN_MODEL.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_prediction = self.class_names[np.argmax(score)]
        print( "This image most likely belongs to {} with a {:.2f} percent confidence.".format(self.class_names[np.argmax(score)], 100 * np.max(score)))
        return class_prediction, 100 * np.max(score)
    
    def crop_bottle_detection(self, image_name, image, width, height):
        cropped_images = []
        image = image
        # image_name = image_name.replace(image_name.split(".")[-1],"jpg")
        try:
            labels = open(YOLO_PREDICTION_LABEL_PATH + "/" + image_name.replace(".jpg", ".txt"), "r").readlines()
            for i, pred in enumerate(labels):
                classes = int(pred.split()[0])
                if classes == 39 : 
                    coordinate = pred.split()
                    print(coordinate)
                    xc, yc, nw, nh = float(coordinate[1]), float(coordinate[2]), float(coordinate[3]), float(coordinate[4])
                    xc  *= width
                    yc *= height
                    nw *= width
                    nh *= height
                    top_left = int(xc-nw/2), int(yc-nh/2)
                    bottom_right = int(xc+nw/2), int(yc+nh/2)
                    crop_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                    image_file_name = image_name.split(".")[0] + f"_cropped_{i}.jpg"
                    cv2.imwrite(image_file_name, crop_img)
                    cropped_images.append([crop_img, image_file_name])
            return cropped_images
        except:
            return []