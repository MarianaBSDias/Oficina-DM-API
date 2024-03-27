import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File

app = FastAPI(docs_url = "/", title = 'Oficina_BI')

# Carregar o pipeline de pré-processamento e inferência
pipeline = joblib.load('breast_pipeline.pkl')

# Criar uma rota para o endpoint
# new*


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Ler o arquivo
    df = pd.read_csv(file.file, index_col = 0)

    # Fazer a previsão
    pred = pipeline.predict(df)
    return {"prediction": pred.tolist()}