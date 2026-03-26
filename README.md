# lab04OA

## Panoramica

Questo repository contiene due script Python di esempio e un file CSV. Il progetto serve come base per esecuzioni locali e sperimentazione.

## File principali

- [ARMA.py](ARMA.py): script principale che legge `BoxJenkins.csv`, applica trasformazioni (log, differenze stagionali e non), e mette a confronto due approcci di modellazione: un modello ARIMA stagionale oppure un modello AutoReg (AR) sui dati trasformati. Effettua previsioni in-sample e forecast a 12 passi, inverte le trasformazioni e mostra i grafici dei risultati.
- [Autoreg.py](Autoreg.py): versione più compatta e procedurale che esegue trasformazioni simili (log, diff(1), diff(12)), adatta un modello AutoReg(2) sui dati trasformati, ricostruisce la serie invertendo le differenze ed esegue la plot di actual / prediction / forecast.
- [BoxJenkins.csv](BoxJenkins.csv): file CSV con la serie temporale (colonna `Passengers`) utilizzata dagli script (es. dataset Box–Jenkins sui passeggeri).

## Requisiti

- Python 3.8 o superiore
- Librerie: `pandas`, `numpy`, `matplotlib`, `statsmodels`

## Esecuzione

Eseguire uno degli script con Python dalla root del progetto:

```bash
python ARMA.py
# oppure
python Autoreg.py
```
