import os
from model_detect_bottle import DetectandClassifyCoke
import cv2
import numpy as np

class ClassificationFLow(DetectandClassifyCoke):
    def __init__(self, image_data):
        self.image = image_data
        super().__init__()  

    def run(self):
        return self.data
    
    def predict(self):
        response = {}
        response["predictions"] = []
        confidence = 0
        prediction = "Other"
        input_image = (cv2.imread("temp_image.jpg"), "temp_image.jpg")
        yolo_detected_bottles = self.get_bottle_detection_with_yolo()
        detected_bottles = yolo_detected_bottles if yolo_detected_bottles else [input_image]
        for i, bottle in  enumerate(detected_bottles):
            classification_result, confidence = self.get_bottle_classification_with_cnn(bottle)
            if classification_result == "Coke" and confidence > 70:
                prediction = "CocaCola"
            elif classification_result == "Nocoke" and confidence > 60 : 
                prediction = "Other"
            response["predictions"].append({f"bottle_{i}" : {"class": prediction, "confidence": confidence}})
        
        all_bottle_predictions = [response["predictions"][i][f"bottle_{i}"]["class"] for i in range(len(response["predictions"]))]
        if "CocaCola" in all_bottle_predictions:
            response["is_cocacola"] = True
        else:
            response["is_cocacola"] = False
        return response