import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA


def difference(data, lag): #faccio la diff 
    return np.array([data[i] - data[i - lag] for i in range(lag, len(data))])
#potevo usare un unica funzione la invert della differenza.
def invert_seasonal_difference12(base_diff, seasonal_diff_pred, lag):
    reconstructed = list(base_diff[:lag])
    for i in range(len(seasonal_diff_pred)):
        reconstructed.append(seasonal_diff_pred[i] + reconstructed[i])
    return np.array(reconstructed)

def invert_first_difference1(original_log, diff_series):
    reconstructed = [original_log[0]]
    for i in range(len(diff_series)):
        reconstructed.append(diff_series[i] + reconstructed[i])
    return np.array(reconstructed)

if __name__ == "__main__":
    # ---- 1. Lettura dati ----
    df = pd.read_csv("BoxJenkins.csv", usecols=["Passengers"])
    y = df["Passengers"].to_numpy()

    print(f"dataset shape: {y.shape}")

    # ---- 2. Log + differenza ----
    y_log = np.log(y)
    y_log_diff1 = difference(y_log, 1)
    y_log_diff12_diff1 = difference(y_log_diff1, 12)
    #stampe di debug per verificare le forme delle serie trasformate
    print(f"log series shape: {y_log.shape}")
    print(f"diff(1) shape: {y_log_diff1.shape}")
    print(f"diff(12) after diff(1) shape: {y_log_diff12_diff1.shape}")

    # Gli ultimi 12 punti della serie trasformata vengono lasciati fuori dal fit
    train_transformed = y_log_diff12_diff1[:-12]

    # ---- 3. Scelta modello: AR(2) vs ARIMA ----
    use_arima = True

    if use_arima:
        # Parametri ARIMA (esempio): order=(p,d,q), seasonal_order=(P,D,Q,s)
        arima_order = (1, 0, 1)
        seasonal_order = (0, 1, 1, 12)

        arima_model = ARIMA(y_log, order=arima_order, seasonal_order=seasonal_order)
        arima_res = arima_model.fit()
        print(arima_res.summary())

        in_sample_pred = arima_res.predict(start=0, end=len(y_log) - 1)
        out_of_sample_forecast = arima_res.forecast(steps=12)
        y_log_pred = np.concatenate([in_sample_pred, out_of_sample_forecast])

        # Torniamo alla scala originale
        y_pred = np.exp(y_log_pred)

        plt.figure("Confronto finale - ARMA", figsize=(10, 6))
        plt.plot(y, label="Actual")
        # Non mostrare i primi 12 punti previsionali (spesso instabili/null)
        plt.plot(range(12, len(y)), y_pred[12:len(y)], label="Prediction")
        plt.plot(range(len(y), len(y_pred)), y_pred[len(y):], "r", label="Forecast")
        plt.legend()
        plt.show()
    else: #se metto a false uso AR(2) sui dati trasformati manualmente
        # ---- 3. Fit modello AR(2) ----
        model = AutoReg(train_transformed, lags=2)
        model_fit = model.fit()
        print(model_fit.summary())

        # ---- 4. Prediction + forecast ----
        in_sample_pred = model_fit.predict(start=0, end=len(train_transformed) - 1)
        out_of_sample_forecast = model_fit.forecast(steps=12)
        y_log_diff12_diff1_pred = np.concatenate([in_sample_pred, out_of_sample_forecast])

        # I primi 2 valori non sono ben definiti dal modello AR(2)
        y_log_diff12_diff1_pred[:2] = y_log_diff12_diff1[:2]

        # ---- 5. Inversione trasformazioni ----
        y_log_diff1_pred = invert_seasonal_difference12(y_log_diff1, y_log_diff12_diff1_pred, 12)
        y_log_pred = invert_first_difference1(y_log, y_log_diff1_pred)
        y_pred = np.exp(y_log_pred)

        plt.figure("Confronto finale", figsize=(10, 6))
        plt.plot(y, label="Actual")
        plt.plot(range(12, len(y)), y_pred[12:len(y)], label="Prediction")
        plt.plot(range(len(y_pred) - 12, len(y_pred)), y_pred[-12:], "r", label="Forecast")
        plt.legend()
        plt.show()