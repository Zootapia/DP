# Advanced Time Series Forecasting with Deep Learning and Attention Mechanisms
This project implements a multivariate time-series forecasting pipeline using a hybrid LSTM-Attention architecture. It compares the performance of this deep learning approach against traditional statistical models (SARIMA) and additive models (Prophet).
# Project Overview
The primary goal is to predict the future values of a target variable (x1) based on its historical values and correlated features (x2, x3). The project includes data generation, preprocessing, model development, and a final benchmark comparison.
# 🏗️ Model Architecture: LSTM + Attention
The core of this project is a custom PyTorch model that combines the sequential processing power of LSTMs with an Attention mechanism to focus on relevant time steps.
LSTM Layer: Captures temporal dependencies and long-term patterns in the multivariate input.
Attention Mechanism: Computes importance weights for each hidden state produced by the LSTM, allowing the model to "attend" to specific past events that are more predictive of the future.
Fully Connected Layer: Maps the weighted context vector to a single scalar prediction.
# 📊 Dataset Description
The model uses synthetically generated multivariate data to simulate real-world complexities:
x1: The target variable, featuring a linear trend, sinusoidal seasonality, and Gaussian noise.
x2: A feature highly correlated with x1.
x3: A cyclical feature representing external seasonal influences.
# 🛠️ Implementation Details
1. Requirements
Deep Learning: torch
Data Handling: numpy, pandas, scikit-learn
Baselines: statsmodels (SARIMA), prophet
Visualization: matplotlib
2. Preprocessing
Scaling: Data is normalized using MinMaxScaler to range between [0, 1].
Windowing: Input data is transformed into sequences of length 30 (SEQ_LEN = 30) to create a supervised learning format.
# 📈 Performance Comparison
The models were evaluated using Root Mean Squared Error (RMSE). The deep learning model significantly outperformed the traditional baselines on this specific dataset.
# Model RMSE
LSTM + Attention 0.0788
SARIMA 1.0442
Prophet 7.6601
# How to Use
Data Generation: Run the generate_time_series() function to create the dataset.
Training: The LSTMAttention model is trained for 20 epochs using the Adam optimizer and MSE loss.
Evaluation: Compare results using the provided evaluation script which calculates MAE, RMSE, and MAPE.
