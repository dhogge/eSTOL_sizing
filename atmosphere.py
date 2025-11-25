from typing import Tuple
import numpy as np

def atmosisa(altitude_m: float) -> Tuple[float, float, float, float]:
    """
    International Standard Atmosphere (ISA) model.
    Returns (T [K], P [Pa], rho [kg/m³], a [m/s]).
    """
    T0, P0, g, R = 288.15, 101325, 9.80665, 287.05
    if altitude_m <= 11000:
        L = -0.0065
        T = T0 + L * altitude_m
        P = P0 * (T / T0) ** (-g / (L * R))
    else:
        T11, P11 = 216.65, 22632.1
        T = T11
        P = P11 * np.exp(-g * (altitude_m - 11000) / (R * T))
    rho = P / (R * T)
    a = np.sqrt(1.4 * R * T)
    return T, P, rho, a
