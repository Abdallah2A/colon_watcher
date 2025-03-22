import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from pydantic import BaseModel
from ultralytics import YOLO
import logging

# Initialize FastAPI app
app = FastAPI(title="Colon Watcher")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv10 model (update the model path accordingly)
try:
    model = YOLO("../saved_models/p_best.pt")  # Use appropriate model variant
    logger.info("model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to initialize model")


class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]


class ErrorResponse(BaseModel):
    error: str
    details: str = None


@app.post("/detect/",
          response_model=List[DetectionResult],
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def detect_objects(
        file: UploadFile = File(..., description="Image file to process")
):
    """
    Process image through YOLOv10 model and return detection results
    """
    try:
        # Validate input file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file."
            )

        # Read and process image
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Invalid or corrupted file."
            )

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model.predict(image, device='gpu')

        # Process results
        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                conf = box.conf.cpu().numpy()[0]
                cls_id = int(box.cls.cpu().numpy()[0])
                class_name = model.names[cls_id]

                detections.append({
                    "class_name": class_name,
                    "confidence": float(conf),
                    "bbox": xyxy
                })

        return JSONResponse(content=detections)

    except HTTPException as he:
        raise he
    except Exception as err:
        logger.error(f"Detection error: {str(err)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "details": str(err)
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000)
