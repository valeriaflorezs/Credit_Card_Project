"""Réplica del preprocesamiento del proyecto (src/preprocessing.py::build_final_dataset + one-hot)
para las pruebas de diseño de CS-FOF. Devuelve X, y y los 5 pliegues externos (StratifiedKFold, semilla 42)."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
XLS = Path(__file__).resolve().parents[2] / "data" / "raw" / "default_of_credit_card_clients.xls"
def slog(x): return np.sign(x) * np.log1p(np.abs(x))
def cargar():
    df = pd.read_excel(XLS, header=1).rename(columns={"default payment next month": "default_payment_next_month"})
    df = df.drop(columns="ID").reset_index(drop=True)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4}); df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    df = df.drop_duplicates().reset_index(drop=True)
    bills = df[[f"BILL_AMT{i}" for i in range(1, 7)]].to_numpy(float)
    t = np.arange(5, -1, -1, dtype=float); tc = t - t.mean()
    df["BILL_AMT_MEAN_6M_LOG"] = slog(bills.mean(1)); df["BILL_AMT_TREND_6M_LOG"] = slog(bills @ tc / (tc @ tc))
    df["N_CREDIT_BALANCE_MONTHS"] = (bills < 0).sum(1)
    df = df.drop(columns=[f"BILL_AMT{i}" for i in range(1, 7)])
    for c in ["LIMIT_BAL"] + [f"PAY_AMT{i}" for i in range(1, 7)]:
        df[c + "_LOG"] = slog(df[c].astype(float)); df = df.drop(columns=c)
    y = df.pop("default_payment_next_month").to_numpy()
    # one-hot igual a OneHotEncoder(drop="if_binary"): SEX binaria -> 1 columna
    Xdf = pd.get_dummies(df, columns=["SEX", "EDUCATION", "MARRIAGE"], dtype=float)
    Xdf = Xdf.drop(columns=["SEX_1"])
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xdf, y))
    return Xdf.to_numpy(float), y, list(Xdf.columns), folds
if __name__ == "__main__":
    X, y, cols, folds = cargar()
    print(X.shape, round(y.mean(), 4), len(cols)); print(cols)
