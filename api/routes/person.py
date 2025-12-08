"""
Rutas para detección de personas
Responsable: Manejar requests HTTP, validar entradas, formatear respuestas
"""
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import logging
import time
from typing import Optional

from api.services.person_service import person_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/persons")
async def predict_person(file: UploadFile = File(...)):
    """
    Detecta personas en una imagen usando YOLOv8
    
    - **file**: Imagen (JPEG, PNG, WebP) -
    
    Returns:
        JSON con personas detectadas o imagen PNG (debug mode)
    """
    start_time = time.time()
    
    try:
        # 1. VALIDACIÓN
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó archivo")
        
        # Validar tipo de archivo
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Formato no soportado. Use: {', '.join(allowed_extensions)}"
            )
        
        # 2. LEER Y PROCESAR IMAGEN
        contents = await file.read()
        
        # Limitar tamaño (10MB)
        #if len(contents) > 10 * 1024 * 1024:
        #    raise HTTPException(status_code=400, detail="Imagen demasiado grande (max 10MB)")
        
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        '''
        # Redimensionar si es muy grande (para rendimiento)
        if max(image.width, image.height) > max_size:
            ratio = max_size / max(image.width, image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            image = image.resize((new_width, new_height))
            logger.info(f"📐 Imagen redimensionada: {new_width}x{new_height}")
        '''
        
        # 3. DETECCIÓN (LLAMA AL SERVICIO)
        detections = person_service.detect_persons(
            image=image,
            confidence_threshold=0.5,
            image_size=640
        )
        
        processing_time = time.time() - start_time
        
           
        
        # Respuesta JSON normal
        return JSONResponse({
                "person_count": len(detections),
                "persons": detections,
                "processing_time_seconds": round(processing_time, 3),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint /predict_persons: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/person/model-info")
async def get_person_model_info():
    """Obtener información del modelo de personas"""
    return person_service.get_model_info()