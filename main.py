from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tomato Disease Detection API")

# CORS untuk Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv11 model
try:
    model = YOLO('best.pt')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

@app.get("/")
async def root():
    return {
        "message": "Tomato Disease Detection API",
        "version": "1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Validasi file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Baca dan proses image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert RGBA ke RGB jika perlu
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Prediksi dengan YOLOv11
        results = model(image, conf=0.25)
        
        # Parse hasil deteksi
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                detections.append({
                    "class_name": result.names[class_id],
                    "confidence": float(box.conf[0]),
                    "bbox": {
                        "x1": float(box.xyxy[0][0]),
                        "y1": float(box.xyxy[0][1]),
                        "x2": float(box.xyxy[0][2]),
                        "y2": float(box.xyxy[0][3])
                    }
                })
        
        # Sorting berdasarkan confidence tertinggi
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            "success": True,
            "total_detections": len(detections),
            "detections": detections,
            "image_size": {
                "width": image.width,
                "height": image.height
            }
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Prediksi multiple images sekaligus"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results_list = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            results = model(image, conf=0.25)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    detections.append({
                        "class_name": result.names[class_id],
                        "confidence": float(box.conf[0])
                    })
            
            results_list.append({
                "filename": file.filename,
                "success": True,
                "total_detections": len(detections),
                "detections": detections
            })
        
        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_images": len(files),
        "results": results_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
