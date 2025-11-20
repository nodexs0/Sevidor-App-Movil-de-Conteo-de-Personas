from roboflow import Roboflow

rf = Roboflow(api_key="EnShkuYpMfrmMLrhD9LX")
project = rf.workspace("patricia-iosif-wp0yg").project("openimages-doors")

# Mostrar versiones disponibles
for v in project.versions():
    print(v.version)

# Cargar versión correcta
model = project.version(2).model

if model is not None:
    result = model.predict("puerta.jpg").json()
    print(result)
else:
    print("No se pudo cargar el modelo. Revisa la versión y la API key.")
