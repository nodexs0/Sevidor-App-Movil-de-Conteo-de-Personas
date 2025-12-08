from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#from api.core.cofig import settings
from api.routes import person, door
import uvicorn

#setup_logging()

app = FastAPI(
    tittle="Deteccion de Personas y Puertas API",
    version="1.0"
)



app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT"],
    allow_headers=["*"],
)

app.include_router(person.router, prefix="/predict", tags=["predicciones"])
app.include_router(door.router, prefix="/predict", tags=["predicciones"])


@app.get("/")
async def root():
    return {"message": "Servidor de detección de personas (Yolov8m) y puertas (Roboflow) funcionando"}



if __name__ == "__main__":
    print("[logs] Iniciando servidor...") 
    uvicorn.run(app, host="0.0.0.0", port=8000)
