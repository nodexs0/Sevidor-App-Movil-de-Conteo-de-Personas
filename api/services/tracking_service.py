"""
Servicio para tracking de personas frame por frame
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from threading import Lock
import cv2

from api.services.session_manager import TrackingSession, session_manager

logger = logging.getLogger(__name__)

class FrameTrackingService:
    """Servicio para tracking frame por frame"""
    
    _instance = None
    _yolo_model = None
    _tracker = None
    _lock = Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True
    
    def _initialize(self):
        """Inicializar YOLO y DeepSORT"""
        try:
            model_path = "api/ia_models/yolov8m.pt"
            
            import os
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo YOLO no encontrado: {model_path}")
            
            # Cargar modelo YOLO
            self._yolo_model = YOLO(model_path)
            
            # Inicializar DeepSORT tracker
            self._tracker = DeepSort(
                max_age=30,
                n_init=1,
                )
            
            logger.info("FrameTrackingService inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando FrameTrackingService: {e}")
            self._yolo_model = None
            self._tracker = None
    
    def centroid(self, l: float, t: float, r: float, b: float) -> Tuple[int, int]:
        """Calcular centroide del bounding box"""
        return int((l + r) / 2), int((t + b) / 2)
    
    def in_door(self, cx: int, cy: int, door_bbox: Tuple) -> bool:
        """Verificar si punto está dentro del área de la puerta"""
        door_x1, door_y1, door_x2, door_y2 = door_bbox
        return door_x1 <= cx <= door_x2 and door_y1 <= cy <= door_y2
    
    def process_frame(
        self,
        session_id: str,
        image: Image.Image,
        run_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Procesar un frame individual para tracking
        
        Args:
            session_id: ID de la sesión activa
            image: Imagen PIL
            run_detection: Si ejecutar YOLO en este frame
        
        Returns:
            Resultados del procesamiento del frame
        """
        if self._yolo_model is None or self._tracker is None:
            raise RuntimeError("Servicio de tracking no inicializado")
        
        # Obtener sesión
        session = session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Sesión no encontrada: {session_id}")
        
        start_time = time.time()
        
        # Convertir PIL a numpy (BGR para OpenCV)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Guardar frame para debug (opcional)
        cv2.imwrite(f"debug_frames/session_{session_id}_frame_{session.frame_count}.jpg", frame)

        detections = []
        yolo_time = 0
        
        # Ejecutar YOLO si es necesario
        if run_detection:
            yolo_start = time.time()
            
            # Detectar personas (solo clase 0)
            results = self._yolo_model(frame, classes=[0], verbose=False)
            
            for b in results[0].boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                w, h = x2 - x1, y2 - y1
                detections.append(([x1, y1, w, h], float(b.conf[0]), "person"))

            # Guardar boxes detectados para debug (opcional)
            for det in detections:
                box, conf, cls = det
                x1, y1, w, h = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.imwrite(f"debug_frames/session_{session_id}_frame_{session.frame_count}_detections.jpg", frame)
            
            yolo_time = time.time() - yolo_start
            session.yolo_calls += 1
            session.yolo_total_time += yolo_time
        
        # Actualizar tracker
        tracks = self._tracker.update_tracks(detections, frame=frame)
        
        current_ids = set()
        current_detections = []
        
        # Procesar tracks
        for tr in tracks:
            if not tr.is_confirmed():
                continue
            
            tid = tr.track_id
            current_ids.add(tid)
            
            l, t, r, b = tr.to_ltrb()
            cx, cy = self.centroid(l, t, r, b)
            is_in_door = self.in_door(cx, cy, session.door_bbox)
            
            current_detections.append({
                "track_id": tid,
                "bbox": [float(l), float(t), float(r), float(b)],
                "centroid": [cx, cy],
                "in_door": is_in_door
            })
            
            if not session.initialized:
                session.tracks_state[tid] = {
                    "in_door": is_in_door,
                    "last_frame_seen": session.frame_count,
                }
                continue
            
            if tid not in session.tracks_state:
                if is_in_door:
                    session.entradas += 1
                
                session.tracks_state[tid] = {
                    "in_door": is_in_door,
                    "last_frame_seen": session.frame_count,
                }
            else:
                session.tracks_state[tid]["in_door"] = is_in_door
                session.tracks_state[tid]["last_frame_seen"] = session.frame_count
        
        # Primera inicialización
        if not session.initialized and len(current_ids) > 0:
            session.initialized = True
            session.iniciales = len(current_ids)
        
        # Procesar salidas
        ids_to_delete = []
        for tid, st in list(session.tracks_state.items()):
            if tid not in current_ids:
                if session.frame_count - st["last_frame_seen"] == session.disappear_buffer:
                    if st["in_door"]:
                        session.salidas += 1
                    ids_to_delete.append(tid)
        
        for tid in ids_to_delete:
            session.tracks_state.pop(tid, None)
        
        # Actualizar contador de frames
        session.frame_count += 1
        
        # Calcular métricas
        processing_time = time.time() - start_time
        personas_dentro = session.iniciales + session.entradas - session.salidas
        
        # Actualizar sesión
        session_manager.update_session(session_id, {
            "frame_count": session.frame_count,
            "entradas": session.entradas,
            "salidas": session.salidas,
            "iniciales": session.iniciales,
            "tracks_state": session.tracks_state,
            "initialized": session.initialized,
            "yolo_calls": session.yolo_calls,
            "yolo_total_time": session.yolo_total_time
        })

        print(f"Processed frame {session.frame_count} of session {session_id}: "
              f"{len(detections)} detections, {len(current_detections)} tracks, "
              f"processing time {processing_time:.3f}s (YOLO time {yolo_time:.3f}s)")
        
        return {
            "session_id": session_id,
            "frame_number": session.frame_count,
            "processing_time_ms": round(processing_time * 1000, 2),
            "yolo_time_ms": round(yolo_time * 1000, 2) if yolo_time > 0 else 0,
            "detections_count": len(detections),
            "tracks_count": len(current_detections),
            "current_detections": current_detections,
            "statistics": {
                "personas_iniciales": session.iniciales,
                "entradas_acumuladas": session.entradas,
                "salidas_acumuladas": session.salidas,
                "personas_dentro_actual": personas_dentro,
                "tracks_activos": len(session.tracks_state)
            },
            "run_detection": run_detection
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        return {
            "loaded": self._yolo_model is not None and self._tracker is not None,
            "yolo_model": "yolov8m.pt",
            "tracker": "DeepSORT",
            "max_age": 30,
            "status": "ready" if self._yolo_model else "error"
        }


# Instancia global
try:
    frame_tracking_service = FrameTrackingService()
except Exception as e:
    logger.error(f"No se pudo inicializar FrameTrackingService: {e}")
    frame_tracking_service = None