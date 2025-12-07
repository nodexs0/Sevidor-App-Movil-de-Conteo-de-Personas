"""
Servicio para detección de personas usando YOLOv8.
Implementa el patrón Singleton para cargar el modelo una sola vez.
"""
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import logging
from typing import List, Dict, Any, Optional
from threading import Lock

# Configuración básica de logging para registrar información y errores
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonDetectorService:
    """
    Servicio singleton para detección de personas.
    Carga el modelo YOLO una sola vez y lo reutiliza para todas las peticiones.
    """
    
    # Variables de clase para el patrón Singleton
    _instance: Optional['PersonDetectorService'] = None
    _lock: Lock = Lock()  # Lock para hacer el Singleton thread-safe
    _model: Optional[YOLO] = None  # Instancia del modelo YOLO
    _initialized: bool = False  # Bandera para controlar la inicialización única
    
    def __new__(cls) -> 'PersonDetectorService':
        """
        Método especial que controla la creación de instancias.
        Implementa el patrón Singleton de forma thread-safe.
        
        Returns:
            La única instancia de PersonDetectorService
        """
        if cls._instance is None:
            with cls._lock:
                # Doble verificación para evitar condiciones de carrera
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """
        Inicializador que se ejecuta solo una vez.
        Carga el modelo cuando se crea la primera instancia.
        """
        if not self._initialized:
            with self._lock:
                # Segunda verificación dentro del lock
                if not self._initialized:
                    self._initialize_model()
                    self._initialized = True
    
    def _initialize_model(self) -> None:
        """
        Carga y valida el modelo YOLO desde el archivo .pt
        Verifica que el archivo exista antes de cargarlo.
        """
        try:
            # Ruta al archivo del modelo entrenado
            model_path = "api/ia_models/yolov8m.pt"
            
            logger.info(f"Inicializando modelo desde: {model_path}")
            
            # Verificar que el archivo del modelo existe
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el archivo del modelo: {model_path}")
            
            # Cargar el modelo YOLO
            self._model = YOLO(model_path)
            
            # Verificar que el modelo se cargó correctamente
            if self._model is None:
                raise RuntimeError("Error al cargar el modelo YOLO")
            
            logger.info("Modelo cargado exitosamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar el modelo: {e}")
            raise
    
    def detect_persons(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.5,
        image_size: int = 640
    ) -> List[Dict[str, Any]]:
        """
        Detecta personas en una imagen usando el modelo YOLO.
        
        Args:
            image: Objeto PIL.Image con la imagen a procesar
            confidence_threshold: Umbral de confianza mínimo (0.0 a 1.0)
            image_size: Tamaño al que se redimensiona la imagen para el modelo
            
        Returns:
            Lista de diccionarios, cada uno con:
                - "bbox": Lista con coordenadas [x1, y1, x2, y2] del bounding box
                - "score": Puntuación de confianza de la detección
            
        Raises:
            RuntimeError: Si el modelo no está inicializado
            RuntimeError: Si ocurre un error durante la detección
        """
        # Verificar que el modelo esté cargado
        if self._model is None:
            raise RuntimeError("El modelo no ha sido inicializado")
        
        # Validar parámetros de entrada
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"El umbral de confianza debe estar entre 0 y 1, se recibió: {confidence_threshold}")
        
        try:
            # Convertir la imagen PIL a array numpy para procesamiento
            img_array = np.array(image)
            
            logger.info(f"Procesando imagen de tamaño: {img_array.shape}")
            
            # Ejecutar predicción con el modelo YOLO
            # imgsz: tamaño de entrada para el modelo
            # conf: umbral de confianza mínimo
            # verbose: desactiva la salida detallada de ultralytics
            results = self._model.predict(
                img_array,
                imgsz=image_size,
                conf=confidence_threshold,
                verbose=False
            )
            
            # Lista para almacenar las detecciones de personas
            persons = []
            
            # Procesar resultados de la predicción
            for result in results:
                # Verificar si hay bounding boxes en el resultado
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    # Iterar sobre cada detección
                    for box, score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                        # Filtrar solo la clase 0 (persona)
                        # IMPORTANTE: Verificar que en tu modelo la clase 0 sea "persona"
                        if int(cls) == 0:
                            # Convertir tensores a listas de Python
                            bbox_list = box.cpu().numpy().tolist() if hasattr(box, 'cpu') else box.tolist()
                            score_float = float(score)
                            
                            # Agregar detección a la lista
                            persons.append({
                                "bbox": bbox_list,
                                "score": score_float
                            })
            
            logger.info(f"Se detectaron {len(persons)} personas en la imagen")
            
            return persons
            
        except Exception as e:
            logger.error(f"Error durante la detección de personas: {e}")
            raise RuntimeError(f"Error en detección de personas: {str(e)}")
    
    def is_ready(self) -> bool:
        """
        Verifica si el servicio está listo para realizar inferencias.
        
        Returns:
            True si el modelo está cargado y listo, False en caso contrario
        """
        return self._model is not None and self._initialized


# Crear instancia global del servicio
# Esta instancia se crea al importar el módulo y se reutiliza en toda la aplicación
try:
    person_service = PersonDetectorService()
except Exception as e:
    logger.error(f"No se pudo crear el PersonDetectorService: {e}")
    # En caso de error, person_service será None
    # Los endpoints deben verificar esto antes de usar el servicio
    person_service = None

