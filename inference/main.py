import pickle
import pandas as pd
import logging
import time

try:
    logging.basicConfig(level=logging.INFO)
    filename = 'model/model.pkl'
    loaded_model = pickle.load(open(filename, 'rb'))
except FileNotFoundError:
    raise Exception("Trained model file not found.")

try:
    df = pd.read_csv('data/inference_dataset.csv')
    logging.info(f"Size of inference dataset: {df.shape}")
except FileNotFoundError:
    raise Exception("Inference dataset file not found.")

try:
    start = time.time();
    predictions = loaded_model.predict(df.drop(['Species', 'Id'], axis=1))
    end = time.time()
    logging.info(f"Inference completed in {end - start} seconds")
except AttributeError:
    raise Exception("Model doesn't have 'predict' method.")

df['Results'] = predictions
df.to_csv('data/inference_dataset.csv', index=False)
for ind in df.index:
    print(df['Species'][ind], df['Results'][ind])
