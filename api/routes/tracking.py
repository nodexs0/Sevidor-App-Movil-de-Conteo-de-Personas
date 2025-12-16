"""
Rutas para tracking de personas frame por frame
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import time
from typing import Optional

from api.services.tracking_service import frame_tracking_service
from api.services.session_manager import session_manager

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/tracking/session/start")
async def start_tracking_session(
    door_x1: int = Query(106, description="Coordenada X1 de la puerta"),
    door_y1: int = Query(258, description="Coordenada Y1 de la puerta"),
    door_x2: int = Query(396, description="Coordenada X2 de la puerta"),
    door_y2: int = Query(819, description="Coordenada Y2 de la puerta"),
    detect_interval: int = Query(5, ge=1, le=30, description="Cada cuántos frames detectar"),
    disappear_buffer: int = Query(10, ge=1, le=30, description="Frames para considerar salida")
):
    """
    Iniciar una nueva sesión de tracking en tiempo real
    
    Retorna un session_id que debe usarse en los siguientes frames
    """
    try:
        door_bbox = (door_x1, door_y1, door_x2, door_y2)
        session_id = session_manager.create_session(
            door_bbox=door_bbox,
            detect_interval=detect_interval,
            disappear_buffer=disappear_buffer
        )
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "message": "Sesión de tracking iniciada",
            "config": {
                "door_bbox": door_bbox,
                "detect_interval": detect_interval,
                "disappear_buffer": disappear_buffer
            }
        })
        
    except Exception as e:
        logger.error(f"Error iniciando sesión: {e}")
        raise HTTPException(status_code=500, detail=f"Error iniciando sesión: {str(e)}")

@router.post("/tracking/frame")
async def process_tracking_frame(
    session_id: str = Query(..., description="ID de sesión activa"),
    file: UploadFile = File(..., description="Frame para procesar"),
    force_detection: bool = Query(True, description="Forzar detección en este frame"),
    background_tasks: BackgroundTasks = None
):
    """
    Procesar un frame en una sesión de tracking existente
    
    - **session_id**: ID obtenido al iniciar la sesión
    - **file**: Imagen del frame (JPEG, PNG, WebP)
    - **force_detection**: Forzar ejecución de YOLO en este frame
    """
    start_time = time.time()
    
    # Validar servicio
    if frame_tracking_service is None:
        raise HTTPException(status_code=503, detail="Servicio de tracking no disponible")
    
    try:
        # Validar archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó archivo")
        
        # Validar tamaño
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB máximo por frame
            raise HTTPException(status_code=400, detail="Frame demasiado grande (max 5MB)")
        
        # Cargar imagen
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        

        # Determinar si ejecutar detección
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Sesión no encontrada o expirada")
        
        run_detection = True
        
        # Procesar frame
        result = frame_tracking_service.process_frame(
            session_id=session_id,
            image=image,
            run_detection=run_detection
        )
        
        total_time = time.time() - start_time
        
        # Agregar tiempo total a la respuesta
        result["total_processing_time_ms"] = round(total_time * 1000, 2)
        
        # Limpieza periódica en background
        if background_tasks and session.frame_count % 100 == 0:
            background_tasks.add_task(session_manager.cleanup_old_sessions)
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        logger.error(f"Error de sesión: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error procesando frame: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@router.get("/tracking/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """
    Obtener estadísticas de una sesión de tracking
    """
    stats = session_manager.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    return JSONResponse(content=stats)

@router.delete("/tracking/session/{session_id}")
async def end_tracking_session(session_id: str):
    """
    Finalizar una sesión de tracking y obtener resultados finales
    """
    stats = session_manager.get_session_stats(session_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Sesión no encontrada")
    
    # Eliminar sesión
    session_manager.delete_session(session_id)
    
    return JSONResponse({
        "success": True,
        "message": "Sesión finalizada",
        "final_stats": stats
    })

@router.get("/tracking/sessions/cleanup")
async def cleanup_sessions():
    """
    Limpiar sesiones inactivas (admin)
    """
    cleaned = session_manager.cleanup_old_sessions()
    return {
        "cleaned_sessions": cleaned,
        "message": f"Se limpiaron {cleaned} sesiones inactivas"
    }

@router.get("/tracking/model-info")
async def get_tracking_model_info():
    """Obtener información del modelo de tracking"""
    if frame_tracking_service is None:
        return {"loaded": False, "status": "service_unavailable"}
    
    return frame_tracking_service.get_model_info()