import statistical_analysis
import numpy as np
import pandas as pd

df=pd.read_csv("../test_results.csv")
ds01 = df["f1_score"].to_numpy()
df=pd.read_excel("../F1_Score_cindy.xlsx")
ds02 = df["F1_Score"].to_numpy()

opts = {
    "ds_01": ds01,
    "ds_01_name": "Dataset de Brandon",
    "ds_02": ds02,
    "ds_02_name": "Dataset de Cindy",
    "alpha": {
        "values": [0.01, np.arange(0.05, 1, 0.05)],
        "P": 0.5,
        "k": 1,
        "sample_size": 10,
        "iterations": 500
    }
}

resultados = statistical_analysis.statistical_analysis(opts)
statistical_analysis.save_results(resultados)
print("Análisis completado. Resultados guardados en statistical_analysis_results.json")
