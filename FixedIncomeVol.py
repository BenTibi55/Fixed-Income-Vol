import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import os

plt.style.use('seaborn-darkgrid')

ROLLING_WINDOW = 21
RISK_FREE_RATE = 0.01

def generate_mock_yield_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='B')
    yields = 0.02 + np.cumsum(np.random.normal(0, 0.0003, len(dates)))
    df = pd.DataFrame({'Date': dates, 'Yield': yields})
    df.set_index('Date', inplace=True)
    df['LogReturn'] = np.log(df['Yield'] / df['Yield'].shift(1))
    df.dropna(inplace=True)
    return df

def calculate_historical_vol(df, window):
    df['HistVol'] = df['LogReturn'].rolling(window=window).std() * np.sqrt(252)
    return df

def plot_historical_vol(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['HistVol'], label='Annualized Realized Volatility')
    plt.title('Historical Volatility (Synthetic Yield Curve)')
    plt.ylabel('Volatility')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figs/historical_volatility.png')

def create_ir_vol_surface():
    strikes = np.linspace(0.01, 0.05, 20)
    tenors = np.linspace(0.25, 5.0, 20)
    K, T = np.meshgrid(strikes, tenors)
    vol_surface = 0.2 + 0.15 * np.abs(K - 0.03) * np.exp(-T / 5)
    return K, T, vol_surface

def plot_vol_surface(K, T, Z):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K, T, Z, cmap='viridis')
    ax.set_title('Synthetic Implied Volatility Surface (IR Options)')
    ax.set_xlabel('Strike Rate')
    ax.set_ylabel('Tenor (Years)')
    ax.set_zlabel('Implied Volatility')
    plt.tight_layout()
    plt.savefig('figs/implied_vol_surface.png')

def black_formula(F, K, T, sigma, option_type='payer'):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'payer':
        return np.exp(-RISK_FREE_RATE * T) * (F * stats.norm.cdf(d1) - K * stats.norm.cdf(d2))
    else:
        return np.exp(-RISK_FREE_RATE * T) * (K * stats.norm.cdf(-d2) - F * stats.norm.cdf(-d1))

def plot_swaption_surface():
    forwards = np.linspace(0.01, 0.05, 40)
    strikes = np.linspace(0.01, 0.05, 40)
    F, K = np.meshgrid(forwards, strikes)
    T = 1.0  # 1-year tenor
    sigma = 0.2
    prices = black_formula(F, K, T, sigma)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(F, K, prices, cmap='inferno')
    ax.set_title("Swaption Price Surface (Black Model)")
    ax.set_xlabel('Forward Rate')
    ax.set_ylabel('Strike Rate')
    ax.set_zlabel('Option Price')
    plt.tight_layout()
    plt.savefig('figs/swaption_price_surface.png')

if __name__ == '__main__':
    if not os.path.exists('figs'):
        os.makedirs('figs')

    df = generate_mock_yield_data()
    df = calculate_historical_vol(df, ROLLING_WINDOW)
    plot_historical_vol(df)

    K, T, Z = create_ir_vol_surface()
    plot_vol_surface(K, T, Z)

    plot_swaption_surface()
