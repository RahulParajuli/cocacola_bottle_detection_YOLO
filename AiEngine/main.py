from fastapi import APIRouter, FastAPI, Request
import requests
from get_prediction import router as get_prediction_router
from fastapi.middleware.cors import CORSMiddleware

import uvicorn 

app = FastAPI()

app.include_router(get_prediction_router)
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)