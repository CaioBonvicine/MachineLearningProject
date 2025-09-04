# Projeto de Machine Learning - Classificação de Renda

Este projeto implementa um pipeline completo de **Machine Learning** para **classificação de renda** com base em variáveis socioeconômicas.  
O fluxo abrange desde a exploração de dados, pré-processamento, treinamento de modelos até a avaliação de desempenho.

## 🚀 Como usar

### 1. Ajustar caminhos (se necessário)
Dependendo do ambiente em que o projeto for executado, pode ser necessário ajustar os caminhos das pastas dentro dos scripts:  
- `AnaliseDeDataSet.py`  
- `MachineLearningMain.py`

### 2. Criar ambiente virtual (recomendado)

```bash
# Criar ambiente virtual
python -m venv venv

# Ativar no Linux/macOS
source venv/bin/activate

# Ativar no Windows
venv\Scripts\activate

# Instalar Dependências
pip install -r requirements.txt

# Executar a análise exploratória
python AnaliseDeDataSet.py

# Treinar e avaliar o modelo
python MachineLearningMain.py
