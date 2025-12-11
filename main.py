from src.data_exploration import data_exploration
from src.preprocessing import preprocessing
from src.modeling import modeling, load_model
import pandas as pd

data_path = "data/bilheteria.csv"
processed_data_path = "data/bilheteria_processado.csv"

data_exploration(data_path)
preprocessing(data_path)
modeling(processed_data_path)
