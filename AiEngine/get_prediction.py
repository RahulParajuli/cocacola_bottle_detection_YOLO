from PIL import Image
from io import BytesIO
import requests
from utils.utils import delete_cache
from fastapi import APIRouter, FastAPI, Request
from classification.classification_process_flow import ClassificationFLow

router = APIRouter()

def accept_url_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    return image

def responseHelper(code, message, data):
    return {
        "code": code,
        "message": message,
        "data": data
    }

@router.post("/api/v1/predict")
async def predict( request: Request):
    final_response = responseHelper(500, "Internal Server Error", None)
    try:
        try:
            form = await request.form()
            if form["image"]:
                input_image = form["image"].file
            detector = ClassificationFLow(input_image)
            result = detector.predict()
            final_response = responseHelper(200, "Success", result) 
        except Exception as e:
            print(e)
            final_response = responseHelper(400, "Image Declined", None)
    except:
        final_response = responseHelper(500, "Internal Server Error", None)
    delete_cache()
    return final_response

@router.post("/api/v1/predict/image")
async def prediction(request: Request):

    final_response = responseHelper(500, "Internal Server Error", None)
    try:
        try:
            data = await request.json()
            if data["image"]:
                image = accept_url_image(data["image"])
                image.save("temp_image.jpg")
                image = open("temp_image.jpg", "rb")
                detector = ClassificationFLow(image)
                result = detector.predict()
                final_response =  responseHelper(200, "Successfully Detected", result)
            else:
                final_response = responseHelper(400, "Image Declined", None)
        except:
            final_response = responseHelper(400, "Not Valid", None)
    except:
        final_response = responseHelper(500, "Internal Server Error", None)
    delete_cache()
    return final_response



