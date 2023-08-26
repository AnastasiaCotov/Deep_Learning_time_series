# Deep_Learning_time_series
Design and implement forecasting models 
Artificial Neural Networks and Deep Learning Course 2021, Politecnico di Milano

Define the problem
In this challenge we are dealing with a multivariate time series task. The goal is to design and implement forecasting models to learn how to exploit past observations in the input sequence to correctly predict the next 864 future sample.
Big Picture
We have different ways to deal with sequences like hidden Markov models, linear dynamic systems. In this challenge we use Recurrent Neural Networks which can lead us to do forecasting in time series task. One of the most common ways to implement RNN is to use LSTM blocks because they can solve the issue of vanishing gradients. In addition to this, we have two different ways to tackle with this challenge:
1. Direct forecasting which means we forecast 864 samples at a time
2. Autoregressive forecasting in which model can forecast less than 864 samples but with the advantage of sliding windowing, at the end it can predict total samples that we want.
![image](https://github.com/AnastasiaCotov/Deep_Learning_time_series/assets/43670516/831f8ce8-b230-47ad-bbba-95948afcf90c)
