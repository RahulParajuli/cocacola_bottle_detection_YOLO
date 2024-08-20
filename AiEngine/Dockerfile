FROM python:3.9
WORKDIR /AiEngine
COPY AiEngine/requirements.txt .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
# CMD ["uvicorn", "AiEnginemain:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
CMD [ "uvicorn", "main:app" , "--host", "0.0.0.0", "--port", "8000", "--reload"]