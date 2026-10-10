# Libreria di h2.ipynb

Estrazione delle 6 funzioni e delle 2 classi definite nel notebook, senza
eseguire il caricamento del CSV, gli esperimenti o i grafici durante l'import.
Il notebook originale non viene modificato.

| Modulo | Responsabilità | Funzioni e classi |
| --- | --- | --- |
| `data.py` | Suddivisione dei dati e contenitori | `split_dataset_train_val_test`, `split_feature_target_columns`, `IOData`, `ModelData` |
| `preprocessing.py` | Riempimento dei valori mancanti su una copia | `fill_nan_values_of_model_data` |
| `linear_regression.py` | Addestramento e previsione | `train_linear_regression`, `predict_linear_regression` |
| `metrics.py` | Valutazione | `RMSE` |

Tutti i nomi sono disponibili anche direttamente dal pacchetto. Dalla radice
del repository, con Python 3.10 o successivo, NumPy e pandas installati:

```python
import pandas as pd

from homework_2.h2_lib import (
    ModelData,
    RMSE,
    fill_nan_values_of_model_data,
    predict_linear_regression,
    train_linear_regression,
)

df = pd.read_csv("homework_2/car_fuel_efficiency_2026.csv")
data = ModelData.from_dataframe(
    df,
    target_columns=["fuel_efficiency_mpg"],
    training_columns=[
        "engine_displacement", "horsepower", "vehicle_weight", "model_year"
    ],
)
filled = fill_nan_values_of_model_data(data, features_dict={"horsepower": 0})
w = train_linear_regression(filled.train.Xe, filled.train.y)
predictions = predict_linear_regression(filled.val.X, w)
score = RMSE(filled.val.y.ravel(), predictions)
```

Con `homework_2` come directory di lavoro, il pacchetto si importa invece con
`from h2_lib import ...`.

L'estrazione conserva firme e comportamento originali: `pct` indica la quota
di ciascuno dei due insiemi di validazione e test; lo split reimposta il seed
globale di NumPy; `Xe` include la colonna di intercetta, mentre la previsione
riceve `X` senza tale colonna. Anche i controlli e la regolarizzazione
dell'intercetta restano quelli del notebook. Il metodo `IOData.to_matrices`
resta un segnaposto (`pass`), come nell'originale.
