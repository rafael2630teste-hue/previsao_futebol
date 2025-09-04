import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import xgboost as xgb
# importar funções auxiliares se houver
from utils.preprocess import preparar_jogo
from utils.monte_carlo import simular_campeonato

# Caminhos
jogos_path = "dados/jogos_futuros.csv"
jogadores_path = "dados/jogadores.csv"
xgb_model_path = "modelos/xgb_futebol.json"
lstm_model_path = "modelos/lstm_futebol.h5"

# Carregar dados
jogos = pd.read_csv(jogos_path)
jogadores = pd.read_csv(jogadores_path)

# Carregar modelos
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(xgb_model_path)
lstm_model = load_model(lstm_model_path)
