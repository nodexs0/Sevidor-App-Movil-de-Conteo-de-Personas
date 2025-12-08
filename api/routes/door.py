"""
Rutas para detección de puertas
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import time
from typing import Optional

from api.services.door_service import door_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("predict_doors")
async def predict_doors(
    file: UploadFile = File(..., description="Imagen para analizar puertas")):
    """
    Detecta puertas en una imagen usando Roboflow
    
    - **file**: Imagen (JPEG, PNG, WebP)
    """
    start_time = time.time()
    
    try:
        # Validaciones similares a /person
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó archivo")
        
        # Leer imagen
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Imagen demasiado grande (max 10MB)")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Detectar puertas (LLAMA AL SERVICIO)
        detections = door_service.detect_doors(
            image=image
        )
        
        processing_time = time.time() - start_time
        
        
        return JSONResponse(
            content={
                "door_count": len(detections),
                "doors": detections,
                "processing_time_seconds": round(processing_time, 3),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint /predict/doors: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/doors/model-info")
async def get_door_model_info():
    """Obtener información del modelo de puertas"""
    return door_service.get_model_info()
