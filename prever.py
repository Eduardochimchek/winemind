import requests
import json

# Dados de entrada para qualidade máxima
input_data_perfect_quality = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

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
        
        # Definir critérios de qualidade
        if predicted_class == 10:
            quality_label = "Muito bom"
            is_good_quality = True
        elif predicted_class >= 7:
            quality_label = "Bom"
            is_good_quality = True
        else:
            quality_label = "Não tão bom"
            is_good_quality = False
        
        print(f"Classe prevista: {predicted_class}")
        print(f"Qualidade do vinho: {quality_label}")
        print(f"É de boa qualidade? {is_good_quality}")
    else:
        print('Erro na requisição:', response.text)
except Exception as e:
    print('Erro durante a requisição:', str(e))
