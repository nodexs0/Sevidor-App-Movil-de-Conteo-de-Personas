"""
Manejo de sesiones de tracking en tiempo real
"""

import uuid
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)

@dataclass
class TrackingSession:
    """Datos de una sesión de tracking activa"""
    session_id: str
    created_at: float
    last_activity: float
    door_bbox: tuple  # (x1, y1, x2, y2)
    detect_interval: int
    disappear_buffer: int
    
    # Estado del tracking
    tracks_state: Dict[int, Dict[str, Any]]  # track_id -> estado
    initialized: bool
    entradas: int
    salidas: int
    iniciales: int
    frame_count: int
    
    # Métricas
    yolo_calls: int
    yolo_total_time: float

class SessionManager:
    """Gestor singleton de sesiones de tracking"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._sessions = {}
                    cls._instance._cleanup_interval = 300  # 5 minutos
                    cls._instance._last_cleanup = time.time()
        return cls._instance
    
    def create_session(
        self,
        door_bbox: tuple = (106, 258, 396, 819),
        detect_interval: int = 5,
        disappear_buffer: int = 10
    ) -> str:
        """Crear una nueva sesión de tracking"""
        session_id = str(uuid.uuid4())
        
        session = TrackingSession(
            session_id=session_id,
            created_at=time.time(),
            last_activity=time.time(),
            door_bbox=door_bbox,
            detect_interval=detect_interval,
            disappear_buffer=disappear_buffer,
            tracks_state={},
            initialized=False,
            entradas=0,
            salidas=0,
            iniciales=0,
            frame_count=0,
            yolo_calls=0,
            yolo_total_time=0.0
        )
        
        with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"Sesión creada: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[TrackingSession]:
        """Obtener una sesión por ID"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_activity = time.time()
            return session
    
    def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar datos de una sesión"""
        with self._lock:
            if session_id not in self._sessions:
                return False
            
            session = self._sessions[session_id]
            for key, value in updates.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            session.last_activity = time.time()
            return True
    
    def delete_session(self, session_id: str) -> bool:
        """Eliminar una sesión"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Sesión eliminada: {session_id}")
                return True
            return False
    
    def cleanup_old_sessions(self, max_age: int = 1800) -> int:
        """Eliminar sesiones inactivas (30 minutos por defecto)"""
        now = time.time()
        to_delete = []
        
        with self._lock:
            for session_id, session in self._sessions.items():
                if now - session.last_activity > max_age:
                    to_delete.append(session_id)
            
            for session_id in to_delete:
                del self._sessions[session_id]
        
        if to_delete:
            logger.info(f"Limpiadas {len(to_delete)} sesiones inactivas")
        
        return len(to_delete)
    
    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de una sesión en formato dict"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        personas_dentro = session.iniciales + session.entradas - session.salidas
        avg_yolo_time = session.yolo_total_time / session.yolo_calls if session.yolo_calls > 0 else 0
        
        return {
            "session_id": session.session_id,
            "frame_count": session.frame_count,
            "duration_seconds": round(time.time() - session.created_at, 2),
            "statistics": {
                "personas_iniciales": session.iniciales,
                "total_entradas": session.entradas,
                "total_salidas": session.salidas,
                "personas_dentro_actual": personas_dentro,
                "tracks_activos": len(session.tracks_state)
            },
            "performance": {
                "yolo_calls": session.yolo_calls,
                "avg_yolo_time_ms": round(avg_yolo_time * 1000, 2)
            },
            "config": {
                "door_bbox": session.door_bbox,
                "detect_interval": session.detect_interval,
                "disappear_buffer": session.disappear_buffer
            }
        }


# Instancia global
session_manager = SessionManager()