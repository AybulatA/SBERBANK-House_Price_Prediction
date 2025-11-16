import requests

url = 'http://localhost:9696/predict'

house = {
    'sub_area': 'juzhnoe_butovo',
    'ecology': 'satisfactory',
    'product_type': 'investment',
    'full_sq': 39.0,
    'kremlin_km': 24.779082,
    'metro_km_walk': 0.735908,
    'school_km': 0.746962,
    'kindergarten_km': 0.078502,
    'railroad_station_avto_min': 6.274963,
    'usdrub': 55.5989,
    'cpi': 490.5
}

response = requests.post(url, json=house)
price_rub = response.json()

print(price_rub)