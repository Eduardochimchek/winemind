# Projeto de Rede Neural com Flask

## Conteúdo

1. **Visão Geral do Projeto**
2. **Instalação**
3. **Estrutura do Projeto**
4. **Como Usar**
5. **API REST**
6. **Licença**

---

## 1. Visão Geral do Projeto

---

## 2. Instalação

```bash
# Clone o repositório:
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

# Instale as dependências:
# É recomendado criar um ambiente virtual antes de instalar as dependências.
pip install -r requirements.txt

# Configuração do ambiente:
# Certifique-se de ter o Python configurado corretamente. Versão recomendada: Python 3.x.
|-- app.py                   # Arquivo principal da aplicação Flask
|-- model.py                 # Implementação da rede neural
|-- requirements.txt         # Lista de dependências Python
|-- Wine_Quality_Data.csv    # Conjunto de dados de qualidade de vinho
|-- README.md                # Documentação do projeto (você está aqui)

# 1. Treinamento da Rede Neural:
# Execute o script `train.py` para treinar a rede neural com o conjunto de dados de qualidade de vinho.
python train.py

# 2. Iniciar o Servidor Flask:
# Depois de treinar a rede neural, inicie o servidor Flask para servir a API REST.
python app.py

### Endpoint `/predict`

- **Método:** POST
- **URL:** `http://localhost:5000/predict`
- **Corpo da Requisição:**

  ```json
  {
    "data": [7.83, 0.56, 0.29, 2.54, 0.09, 15.9, 46.47, 0.9967, 3.31, 0.66, 10.42]
  }

  {
    "predicted_class": 6,
    "quality_label": "Bom"
  }
  ```

Este README fornece uma visão geral do projeto, instruções de instalação, estrutura do projeto, como usar a rede neural e a API REST, além de informações sobre a licença. Certifique-se de personalizar as seções conforme necessário para o seu projeto específico.