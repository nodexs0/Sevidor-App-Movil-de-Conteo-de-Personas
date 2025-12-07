"""
Servicio para detección de puertas usando Roboflow
"""

import os
import logging
from typing import List, Dict, Any
from PIL import Image
import numpy as np
from roboflow import Roboflow
from threading import Lock



logger = logging.getLogger(__name__)

class DoorDetectionService:
    """Servicio para detección de puertas con Roboflow"""
    
    _instance = None
    _model = None
    _lock = Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DoorDetectionService, cls).__new__(cls)    
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._initialize()
                    self._initialized = True

    
    def _initialize(self):
        """Inicializar modelo de puertas"""
        try:
            # Tu configuración de Roboflow (MANTÉN ESTO SEGURO)
            api_key = "EnShkuYpMfrmMLrhD9LX"
            workspace = "patricia-iosif-wp0yg"
            project_name = "openimages-doors"
            version = 2
            
            logger.info(f"Conectando a Roboflow: {workspace}/{project_name}")
            
            # Inicializar Roboflow
            rf = Roboflow(api_key=api_key)
            
            # Obtener proyecto y modelo
            project = rf.workspace(workspace).project(project_name)
            self._model = project.version(version).model
            
            logger.info("Modelo de puertas configurado correctamente")
            
        except Exception as e:
            logger.error(f"Error configurando Roboflow: {e}")
            self._model = None
    
    def detect_doors(
        self, 
        image: Image.Image,
        confidence_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Detecta puertas en una imagen usando Roboflow
        
        Args:
            image: Imagen PIL
            confidence_threshold: Umbral de confianza
        
        Returns:
            Lista de detecciones de puertas
        """
        # Modo demo si no hay modelo
        if self._model is None:
            raise RuntimeError("El modelo no se cargo correctamente")
        
        try:
            # Convertir PIL a numpy array
            image_array = np.array(image)
            
            #logger.debug(f"Procesando imagen para puertas: {image_array.shape}")
            
            # Hacer predicción con Roboflow
            result = self._model.predict(image_array).json()
            
            # Procesar resultados
            detections = []
            for prediction in result.get("predictions", []):
                # Roboflow usa formato centro + ancho/alto
                x_center = prediction["x"]
                y_center = prediction["y"]
                width = prediction["width"]
                height = prediction["height"]
                # Convertir a formato [x1, y1, x2, y2]
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                    
                detections.append({
                    "bbox": [x_min, y_min, x_max, y_max],
                    "score": prediction["confidence"],
                    "class": prediction["class"],
                })
            
            logger.info(f"Puertas detectadas: {len(detections)}")
            return detections

            
        except Exception as e:
            logger.error(f"Error en detección de puertas: {e}")
            raise RuntimeError(f"Error en detección de puertas: {str(e)}")

    
    def get_model_info(self) -> Dict[str, Any]:
        """Obtener información del modelo"""
        if self._model is None:
            return {
                "loaded": False,
                "mode": "demo",
                "model_name": "Roboflow Doors (Demo Mode)"
            }
        
        return {
            "loaded": True,
            "mode": "production",
            "model_name": "Roboflow OpenImages Doors",
            "workspace": "patricia-iosif-wp0yg",
            "version": 2
        }

# Instancia global del servicio
# Crear instancia global del servicio
# Se inicializa al importar el módulo
try:
    door_service = DoorDetectionService()
except Exception as e:
    logger.error(f"No se pudo inicializar DoorDetectionService: {e}")
    door_service = None