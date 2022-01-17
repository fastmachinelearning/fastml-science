from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from fxpmath import Fxp
import numpy as np


def load_data(bits=None, int_bits=None):
    data = fetch_openml("hls4ml_lhc_jets_hlf")
    X, y = data["data"], data["target"]

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y, 5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if bits and int_bits:
        # cast to fixed point (n_word=16, n_int=5, signed=True overflow=wrap, rounding=floor is closest approximation to Vivado HLS ap_fixed<16,6>)
        X_train = Fxp(
            X_train,
            signed=True,
            n_word=bits,
            n_int=int_bits,
            overflow="wrap",
            rounding="floor",
        )
        X_test = Fxp(
            X_test,
            signed=True,
            n_word=bits,
            n_int=int_bits,
            overflow="wrap",
            rounding="floor",
        )
        # convert back to numpy array
        X_train = np.array(X_train, np.float64)
        X_test = np.array(X_test, np.float64)

    return X_train, X_test, y_train, y_test, le.classes_
