from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import person, door, tracking
import uvicorn

app = FastAPI(
    title="Detección y Tracking de Personas API",
    version="1.0",
    description="""
    API para detección de personas y puertas, con tracking en tiempo real.
    
    ## Endpoints principales:
    
    ### Tracking en tiempo real (frame por frame):
    1. **POST /predict/tracking/session/start** - Iniciar sesión
    2. **POST /predict/tracking/frame** - Procesar frame
    3. **GET /predict/tracking/session/{session_id}/stats** - Ver estadísticas
    4. **DELETE /predict/tracking/session/{session_id}** - Finalizar sesión
    
    ### Detección estática:
    5. **POST /predict/persons** - Detectar personas en imagen
    6. **POST /predict/doors** - Detectar puertas en imagen
    """
)

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(person.router, prefix="/predict", tags=["Detección Estática"])
app.include_router(door.router, prefix="/predict", tags=["Detección Estática"])
app.include_router(tracking.router, prefix="/predict", tags=["Tracking Tiempo Real"])

@app.get("/")
async def root():
    return {
        "message": "API de detección y tracking de personas funcionando",
        "documentation": "/docs",
        "endpoints": {
            "tracking": {
                "start_session": "POST /predict/tracking/session/start",
                "process_frame": "POST /predict/tracking/frame",
                "session_stats": "GET /predict/tracking/session/{id}/stats",
                "end_session": "DELETE /predict/tracking/session/{id}"
            },
            "detection": {
                "persons": "POST /predict/persons",
                "doors": "POST /predict/doors"
            }
        }
    }

if __name__ == "__main__":
    print("[logs] Iniciando servidor con tracking en tiempo real...") 
    uvicorn.run(app, host="0.0.0.0", port=8000)