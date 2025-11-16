# -*- coding: utf-8 -*-

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# -------------------------
# Allowed categories
# -------------------------
ALLOWED_SUB_AREAS = {
    "poselenie_sosenskoe", "nekrasovka", "poselenie_vnukovskoe", "poselenie_moskovskij",
    "poselenie_voskresenskoe", "tverskoe", "mitino", "krjukovo", "poselenie_filimonkovskoe",
    "mar'ino", "poselenie_shherbinka", "juzhnoe_butovo", "solncevo", "zapadnoe_degunino",
    "poselenie_desjonovskoe", "otradnoe", "nagatinskij_zaton", "nagornoe", "strogino",
    "izmajlovo", "tekstil'shhiki", "bogorodskoe", "ljublino", "severnoe_tushino", "gol'janovo",
    "chertanovo_juzhnoe", "birjulevo_vostochnoe", "vyhino-zhulebino", "ochakovo-matveevskoe",
    "zjuzino", "horoshevo-mnevniki", "perovo", "ramenki", "jasenevo", "bibirevo",
    "kosino-uhtomskoe", "golovinskoe", "kuz'minki", "kon'kovo", "caricyno", "veshnjaki",
    "orehovo-borisovo_juzhnoe", "koptevo", "akademicheskoe", "orehovo-borisovo_severnoe",
    "chertanovo_severnoe", "danilovskoe", "novogireevo", "mozhajskoe", "chertanovo_central'noe",
    "ivanovskoe", "obruchevskoe", "pechatniki", "brateevo", "kuncevo", "sokolinaja_gora",
    "presnenskoe", "rjazanskij", "severnoe_butovo", "losinoostrovskoe", "hovrino", "juzhnoe_tushino",
    "dmitrovskoe", "taganskoe", "pokrovskoe_streshnevo", "severnoe_medvedkovo", "beskudnikovskoe",
    "teplyj_stan", "severnoe_izmajlovo", "troickij_okrug", "cheremushki", "nagatino-sadovniki",
    "shhukino", "vostochnoe_izmajlovo", "poselenie_novofedorovskoe", "timirjazevskoe",
    "preobrazhenskoe", "novo-peredelkino", "poselenie_pervomajskoe", "lomonosovskoe",
    "filevskij_park", "kotlovka", "juzhnoe_medvedkovo", "novokosino", "horoshevskoe",
    "donskoe", "levoberezhnoe", "fili_davydkovo", "vojkovskoe", "sviblovo", "juzhnoportovoe",
    "ajeroport", "troparevo-nikulino", "zjablikovo", "lianozovo", "babushkinskoe", "lefortovo",
    "mar'ina_roshha", "jaroslavskoe", "vostochnoe_degunino", "birjulevo_zapadnoe", "krylatskoe",
    "prospekt_vernadskogo", "matushkino", "basmannoe", "silino", "alekseevskoe",
    "moskvorech'e-saburovo", "butyrskoe", "savelki", "meshhanskoe", "hamovniki", "staroe_krjukovo",
    "marfino", "savelovskoe", "gagarinskoe", "ostankinskoe", "jakimanka", "nizhegorodskoe",
    "sokol", "kurkino", "sokol'niki", "altuf'evskoe", "begovoe", "rostokino", "metrogorodok",
    "dorogomilovo", "zamoskvorech'e", "kapotnja", "vnukovo", "krasnosel'skoe", "severnoe",
    "poselenie_rjazanovskoe", "poselenie_rogovskoe", "poselenie_krasnopahorskoe", "poselenie_mosrentgen",
    "poselenie_kokoshkino", "arbat", "vostochnoe", "poselenie_voronovskoe", "poselenie_marushkinskoe",
    "molzhaninovskoe", "poselenie_shhapovskoe", "poselenie_kievskij", "poselenie_mihajlovo-jarcevskoe"
}

ALLOWED_ECOLOGY = {"poor", "no_data", "good", "excellent", "satisfactory"}
ALLOWED_PRODUCT_TYPE = {"investment", "owneroccupier"}

# -------------------------
# Input schema
# -------------------------
class House(BaseModel):
    sub_area: str
    ecology: str
    product_type: str

    full_sq: float = Field(..., ge=0)
    kremlin_km: float = Field(..., ge=0)
    metro_km_walk: float = Field(..., ge=0)
    school_km: float = Field(..., ge=0)
    kindergarten_km: float = Field(..., ge=0)
    railroad_station_avto_min: float = Field(..., ge=0)
    usdrub: float = Field(..., ge=0)
    cpi: float = Field(..., ge=0)

# -------------------------
# Output schema
# -------------------------
class PredictResponse(BaseModel):
    predicted_price_rub: float

# -------------------------
# Load model artifacts
# -------------------------
artifacts = joblib.load('models/model_artifacts.joblib')

# -------------------------
# Prediction function
# -------------------------
def predict_house(house_dict):

    house_df = pd.DataFrame([house_dict])

    cat_encoded = artifacts['encoder'].transform(house_df[artifacts['categorical_col']])

    encoded_df = pd.DataFrame(cat_encoded, columns=artifacts['new_column_names'])

    encoded_df[artifacts['numerical_col']] = house_df[artifacts['numerical_col']].values

    X_single = encoded_df[artifacts['feature_names']].values

    dtest_single = xgb.DMatrix(X_single, feature_names=artifacts['feature_names'])

    price = np.expm1(artifacts['model'].predict(dtest_single)[0])
    
    return float(price)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="house-price-prediction")

# -------------------------
# Endpoint with strict validation
# -------------------------
@app.post("/predict")
def predict(house: House) -> PredictResponse:
    if house.sub_area not in ALLOWED_SUB_AREAS:
        raise HTTPException(status_code=400, detail=f"Invalid sub_area '{house.sub_area}'")
    if house.ecology not in ALLOWED_ECOLOGY:
        raise HTTPException(status_code=400, detail=f"Invalid ecology '{house.ecology}'")
    if house.product_type not in ALLOWED_PRODUCT_TYPE:
        raise HTTPException(status_code=400, detail=f"Invalid product_type '{house.product_type}'")

    price = predict_house(house.model_dump())
    return PredictResponse(predicted_price_rub=price)

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
