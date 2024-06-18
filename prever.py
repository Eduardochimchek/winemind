import requests
import json
import numpy as np
import joblib

# Dados de entrada para qualidade máxima
input_data_perfect_quality = [9.0, 0.3, 0.4, 2.0, 0.05, 20.0, 50.0, 1.0, 3.5, 0.8, 12.0, 10]

# URL da sua API Flask
url = 'http://localhost:5000/predict'

# Dados a serem enviados na requisição POST
data = json.dumps({'data': input_data_perfect_quality})
headers = {'Content-Type': 'application/json'}

try:
    # Enviar a requisição POST e obter a resposta
    response = requests.post(url, headers=headers, data=data)
    
    # Verificar o status da resposta
    if response.status_code == 200:
        result = response.json()
        predicted_class = result['predicted_class']
        quality_label = result['quality_label']
        
        print(f"Classe prevista: {predicted_class}")
        print(f"Qualidade do vinho: {quality_label}")
    else:
        print('Erro na requisição:', response.text)
except Exception as e:
    print('Erro durante a requisição:', str(e))
