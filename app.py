from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from model import Network, DenseLayer  # Importe seu modelo aqui

app = Flask(__name__)

# Função para carregar os dados
def get_data(path, wine_color=None):
    data = pd.read_csv(path)

    if wine_color:
        data = data[data['color'] == wine_color]

    cols = list(data.columns)
    target = cols.pop()

    X = data[cols].copy()
    y = data[target].copy()

    # Transformar rótulos de saída para escala de 0 a 9
    y = LabelEncoder().fit_transform(y)
    # Ajustar para escala de 1 a 10
    y = y + 1

    return np.array(X), np.array(y)

# Carregar dados e normalizar usando StandardScaler
X, y = get_data("Wine_Quality_Data.csv", wine_color='red')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar dados em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Configuração do modelo
model = Network()
model.add(DenseLayer(64, activation='relu'))
model.add(DenseLayer(32, activation='relu'))
model.add(DenseLayer(11, activation='softmax', l2_reg=0.1))  # Ajuste para 11 classes (0 a 10)

# Configurações para o treinamento
epochs = 200  # Número de épocas
learning_rates = [0.001, 0.01, 0.1]  # Diferentes taxas de aprendizado para testar
min_precision = 0.8  # Precisão mínima para parada antecipada

# Loop sobre diferentes taxas de aprendizado
for lr in learning_rates:
    print(f"Treinando modelo com taxa de aprendizado: {lr}")
    model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, epochs=epochs, lr=lr, min_precision=min_precision)

    # Avaliação final do modelo após treinamento
    final_accuracy = model.accuracy[-1] * 100
    final_precision = model.precision[-1] * 100
    print(f"Taxa de aprendizado: {lr}, Precisão final: {final_precision:.2f}%")

    # Verifique se atingiu a precisão mínima
    if final_precision >= min_precision:
        print(f"Modelo treinado com sucesso com precisão mínima de {min_precision * 100}%.")
        break
    else:
        print(f"Modelo não atingiu a precisão mínima de {min_precision * 100}%. Tentando próxima taxa de aprendizado...")

# Rota para previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['data'])  # Acesse corretamente 'data'

        app.logger.info('Dados recebidos: %s', input_data)  # Log dos dados de entrada

        # Verifique se o número de características na entrada corresponde aos dados de treinamento
        if len(input_data) != X_train.shape[1]:  # X_train é a matriz de características usada para treinamento
            raise ValueError("Número de características na entrada não corresponde aos dados de treinamento.")

        # Normalize os dados usando o scaler ajustado nos dados de treinamento
        input_data_scaled = scaler.transform(input_data.reshape(1, -1))

        # Realize previsões
        predictions = model.predict(input_data_scaled)
        predicted_class = int(np.argmax(predictions, axis=1)) + 1  # Converta para escala de 1 a 10

        # Defina critérios de qualidade
        if predicted_class == 10:
            quality_label = "Muito bom"
        elif predicted_class >= 7:
            quality_label = "Bom"
        else:
            quality_label = "Não tão bom"

        app.logger.info('Classe prevista: %s', predicted_class)  # Log da classe prevista

        # Retorne o resultado
        result = {'predicted_class': predicted_class, 'quality_label': quality_label}
        return jsonify(result)
    
    except Exception as e:
        app.logger.error('Erro durante a previsão: %s', str(e))
        return jsonify({'error': 'Erro durante a previsão: ' + str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
