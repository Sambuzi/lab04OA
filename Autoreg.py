import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

if __name__ == "__main__":
	y = pd.read_csv("BoxJenkins.csv", usecols=["Passengers"]).values.flatten()
	ylog = np.log(y)
	ylogdiff = np.diff(ylog)  # or [ylog[i]-ylog[i-1] for i in range(1,len(ylog))]
	# use numpy arrays for consistent broadcasting and slicing
	ylogdiff12 = np.array([ylogdiff[t] - ylogdiff[t - 12] for t in range(12, len(ylogdiff))])
	# fit AR(2) model
	model = AutoReg(ylogdiff12[:-12], lags=2)
	model_fit = model.fit()
	print(model_fit.summary())
	# predict: data from t=0 BOTH ENDS INCLUDED, forecast: only future data
	ylogdiff12_pred = model_fit.predict(0, len(ylogdiff12) - 1)
	# prediction: replace initial p nans
	ylogdiff12_pred[:2] = ylogdiff12[:2]
	# Invert diff(12) -- concatenate the first 12 values with the predicted diffs
	ylogdiff_pred = np.concatenate((ylogdiff[:12], ylogdiff12_pred))
	for i in range(12, len(ylogdiff_pred)):
		ylogdiff_pred[i] += ylogdiff_pred[i - 12]
	# Invert diff(1)
	ylog_pred = np.concatenate(([ylog[0]], ylogdiff_pred))
	for i in range(1, len(ylog_pred)):
		ylog_pred[i] += ylog_pred[i - 1]
	ypred = np.exp(ylog_pred)
	plt.figure(figsize=(10, 6))
	plt.plot(y, 'y', label="Actual")
	plt.plot(ypred[:-12], label="Prediction")
	plt.plot(range(len(ypred) - 12, len(ypred)), ypred[-12:], 'r', label="Forecast")
	plt.legend()
	plt.show()