Stock Price Prediction Pipeline

This repository implements a streamlined and optimized pipeline to forecast the 10-day ahead returns for multiple stocks, based on the Stock Price Prediction Challenge on Kaggle
.

Overview

The goal of this project is to predict future stock returns using historical market data enriched with technical indicators, market features, and temporal encodings.
The pipeline is designed for efficiency and leverages an ensemble of models to maximize prediction accuracy.

Key features

Feature engineering:

Price, volume, momentum, volatility

Technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR)

Market features (dispersion, rolling beta, aggregate indices)

Temporal features (day, month, cyclical encoding, lags, cumulative returns)

Models:

TabNet for multi-output regression

LightGBM (10 separate regressors, one per forecast horizon)

Custom LSTM with attention, trained with mixed precision for speed

Ensembling:
Predictions from all models are combined using validation-based dynamic weights (inverse MSE).

Outputs:

streamlined_submission.csv — ready for Kaggle submission

predictions_plot.png — visualization of 10-day predictions per stock

Requirements

Install dependencies with:

pip install pytorch-tabnet lightgbm ta torch numpy pandas matplotlib scikit-learn joblib


Python 3.8+ is recommended.

Data Structure

Expected directory layout:

input/
  stock-price-prediction-challenge/
    test/
      test_1.csv
      test_2.csv
      ...
    train/indices/
      Dow_Jones.csv
      NASDAQ.csv
      SP500.csv
    sample_submission.csv

Usage

Run the pipeline with:

python pipeline.py


This will train the models, generate ensemble predictions, create the submission file, and save the plot of results.

Example Results

Below is an example visualization of predicted 10-day returns for the five stocks:
