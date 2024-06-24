# Rede Neural para Previsão da Qualidade de Vinho

Este é um projeto de uma aplicação Flask que utiliza uma rede neural para prever a qualidade do vinho com base em suas características. O modelo foi treinado utilizando dados de qualidade de vinho vermelho e implementa várias camadas densas com ativações diferentes.

### Funcionalidades do Projeto

- **Treinamento do Modelo**: A rede neural é treinada com dados de qualidade de vinho vermelho para prever a qualidade em uma escala de 1 a 10.
- **Previsão**: Após o treinamento, a aplicação pode receber dados de entrada sobre as características do vinho e prever sua qualidade.
- **Salvamento de Modelos e Scalers**: O modelo treinado e o scaler usado para normalização dos dados são salvos para uso posterior.

### Estrutura do Projeto

- **`app.py`**: Contém o código principal da aplicação Flask, incluindo a definição das rotas e o código para fazer previsões com o modelo treinado.
- **`model.py`**: Implementa a estrutura da rede neural (`Network` e `DenseLayer`) utilizada para treinar e fazer previsões.
- **`Wine_Quality_Data.csv`**: Dataset utilizado para treinamento da rede neural, contendo informações sobre características e qualidade do vinho.

### Pré-requisitos

- Python 3.9 ou superior
- Pacotes Python necessários estão listados no arquivo `requirements.txt`. Instale-os usando `pip install -r requirements.txt`.

### Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/seu-usuario/nome-do-repositorio.git
   cd nome-do-repositorio

2 - Instale os requisitos:

    pip install -r requirements.txt

### Uso

1- Inicie o servidor Flask:

    python app.py

2 - Faça uma requisição POST para http://localhost:5000/predict com os dados do vinho em formato JSON. Exemplo:

    {
        "data": [9.0, 0.3, 0.4, 2.0, 0.05, 20.0, 50.0, 1.0, 3.5, 0.8, 12.0, 10]
    }

3 - Receba a previsão de qualidade do vinho.

### Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias ou correções.

### Licença

Este projeto está licenciado sob a MIT License.