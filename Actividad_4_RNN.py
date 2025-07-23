# Realizado por Oriana Moreno

import sys
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def rnn(data, scaler, window_sizes):
    for window_size in window_sizes:
        # Crear secuencias
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Modelo
        model = Sequential([
            SimpleRNN(50, activation='relu', input_shape=(window_size, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=100, verbose=0)

        # Prediccion
        y_pred = model.predict(X)
        y_real = scaler.inverse_transform(y)
        y_pred = scaler.inverse_transform(y_pred)

        # Metricas
        mse = np.mean((y_real - y_pred)**2)
        mae = mean_absolute_error(y_real, y_pred)
        mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
        r2 = r2_score(y_real, y_pred)


        # Mostrar por pantalla las metricas
        print(f"Resultados para ventana = {window_size}")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2 * 100:.2f}%")

        graphics(y_real, y_pred, window_size)

def graphics(y_real, y_pred, window_size):
    plt.figure(figsize=(12, 5))
    plt.plot(y_real, 'po-', label='Real')
    plt.plot(y_pred, 'gx--', label='Predicted')
    plt.title(f'Real vs Predicho (ventana = {window_size})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    try:
        df = pd.read_csv("Country-data.csv")
    except FileNotFoundError:
        print("El archivo Country-data.csv no fue encontrado")
        sys.exit(1)

    values = df[['health']].values
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)
    rnn(scaled_values, scaler, window_sizes=[1, 3, 5, 10])


if __name__ == "__main__":
    main()