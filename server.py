from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw
import io
import numpy as np
import time
import os
from ultralytics import YOLO
from roboflow import Roboflow
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Crear carpetas de debug
# -----------------------------
os.makedirs("debug_inputs", exist_ok=True)
os.makedirs("debug_outputs", exist_ok=True)

# -----------------------------
# Cargar modelos
# -----------------------------
print(f"[logs] Cargando modelo de personas")
person_model = YOLO("Modelo/yolov8m.pt")
print("[logs] Modelo de personas cargado correctamente")

print("[logs] Cargando modelo de puertas")
rf = Roboflow(api_key="EnShkuYpMfrmMLrhD9LX")
project = rf.workspace("patricia-iosif-wp0yg").project("openimages-doors")
door_model = project.version(2).model
print("[logs] Modelo de puertas cargado correctamente")

# -----------------------------
# Endpoint: detección de personas
# -----------------------------
@app.post("/predict_persons")
async def predict_person(file: UploadFile = File(...), debug: bool = Query(False)):
    print(f"--"*20)
    print(f"[logs] Prediccion de persona recibida: {file.filename} (debug={debug})")
    try:
        print(f"[logs] Leyendo imagen...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        if debug:
            image.save(os.path.join("debug_inputs", file.filename))

        print(f"[model] Realizando predicción de personas...")
        person_results = person_model.predict(image_array, imgsz=640, conf=0.5)
        persons = []
        for r in person_results:
            for box, score, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                if int(cls) == 0:
                    persons.append({"bbox": box.tolist(), "score": float(score)})
        print(f"[model] Predicción completada. {len(persons)} personas detectadas. data={persons}")

        if debug:
            draw = ImageDraw.Draw(image)
            for p in persons:
                draw.rectangle(p["bbox"], outline="red", width=3)
            out_path = os.path.join("debug_outputs", file.filename)
            image.save(out_path)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        return JSONResponse({"person_count": len(persons), "persons": persons})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# -----------------------------
# Endpoint: detección de puertas
# -----------------------------
@app.post("/predict_doors")
async def predict_doors(file: UploadFile = File(...), debug: bool = Query(False)):
    print(f"--"*20)
    print(f"[logs] Prediccion de puertas recibida: {file.filename} (debug={debug})")
    try:
        print(f"[logs] Leyendo imagen...")
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)

        if debug:
            image.save(os.path.join("debug_inputs", file.filename))

        # Predicción Roboflow online
        print(f"[model] Realizando predicción de puertas...")
        result = door_model.predict(image_array).json()
        doors = []
        for p in result["predictions"]:
            x_min = p["x"] - p["width"]/2
            y_min = p["y"] - p["height"]/2
            x_max = p["x"] + p["width"]/2
            y_max = p["y"] + p["height"]/2
            doors.append({"bbox": [x_min, y_min, x_max, y_max],
                          "score": p["confidence"],
                          "class": p["class"]})
        print(f"[model] Predicción completada. {len(doors)} puertas detectadas. data={doors}")

        if debug:
            draw = ImageDraw.Draw(image)
            for d in doors:
                draw.rectangle(d["bbox"], outline="blue", width=3)
            out_path = os.path.join("debug_outputs", file.filename)
            image.save(out_path)
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/png")

        return JSONResponse({"door_count": len(doors), "doors": doors})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Servidor de detección de personas (Yolov8m) y puertas (Roboflow) funcionando"}

if __name__ == "__main__":
    print("[logs] Iniciando servidor...") 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
