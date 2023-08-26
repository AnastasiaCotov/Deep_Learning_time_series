# Deep_Learning_time_series
Design and implement forecasting models 
Artificial Neural Networks and Deep Learning Course 2021, Politecnico di Milano

Define the problem
In this challenge, we are dealing with a multivariate time series task. The goal is to design and implement forecasting models to learn how to exploit past observations in the input sequence to correctly predict the next 864 future samples.
Big Picture
We have different ways to deal with sequences like hidden Markov models, and linear dynamic systems. In this challenge, we use Recurrent Neural Networks which can lead us to do forecasting in time series tasks. One of the most common ways to implement RNN is to use LSTM blocks because they can solve the issue of vanishing gradients. In addition to this, we have two different ways to tackle this challenge:
1. Direct forecasting which means we forecast 864 samples at a time
2. Autoregressive forecasting in which the model can forecast less than 864 samples but with the advantage of sliding windowing, at the end it can predict the total samples that we want.
![image](https://github.com/AnastasiaCotov/Deep_Learning_time_series/assets/43670516/831f8ce8-b230-47ad-bbba-95948afcf90c)
Results
We started solving the challenge using Conv1D layers with bi-LSTMs which performed well during the training, with a loss below 0.05 and a quite good prediction of time series represented below in plots (prediction is the green line). On the hidden test set this model got RMSE=11.33 which by tunning parameters did not go below 8.3. Here we encountered overfitting.
Our best model during this challenge turns out to be very simple, As we mentioned before our base model consists of a few bi-LTSM layers. As we can see from the pictures below during the training, the model performs very well with time series prediction. Mean square error goes below 0.02 and during testing on the hidden set this model outperformed all the other models we have tried, having RMSE=3.63.

Conclusion
For the given task in challenge 2, we conclude that using more complex models with both Conv1D and bi-LSTM layers learnt to provide good performance during training time, but it gave us poor results during testing on the hidden set, so we ended up with overfitting. One of the reasons this happens is because of the huge number of parameters we had to train with our model. To overcome this issue, we tuned hyperparameters and used early stopping, and dropout layers ending up with a result of 8.3 RMSE.
Moving forward to less complicated models, implementing only bi-LSTM layers we avoid overfitting we took the same strategy as before: multiple tunning of hyperparameters, using early stopping, and regularization. Our best result is 3.6 RMSE.
Solving this challenge made us conclude that there is no best strategy for all types of time series prediction. The most complex model with a lot of layers does not guarantee a better performance. The best way to deal with the problem is to analyze the given dataset, try different combinations of layers in the model, tune hyperparameters and, of course, use techniques to avoid overfitting or vanishing gradient issues.
