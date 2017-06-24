# Data set description 
The data set is available on Kaggle for download - https://www.kaggle.com/dalpozz/creditcardfraud

In summary, it contains 284,807 credit card transactions over 48 hours. It totally contains two types of transactons - fraud and genuine - inside the label column named 'Class'

It is pretty unbalanced - around 0.17% are fraud and the rest are all genuine. The challenge is to build a decent model that is able to tell which are fraud transactions based on the given features of around 30 in total. 

The data set is split based on time series column (unit is in second). The earlier 75% are used as training and validation data, whereas the last 25% are test. AUC (Area Under the ROC Curve) is used as evaluation metric to measure the performance of models. Scores of around 0.95 are reported for our models on test data set. 

# Auto-encoder model 
We are using auto-encoder for unsupervised training - namely, we train auto-encoder on all training data without labelling. 

Also, we tried to explore using auto-encoder as a data pre-processing step, which means we embed our data using auto-encoder's hidden layer, and feed the embeddings as input to a normal feed-forward neural network of binary classification problem. We noticed a slight increase in AUC after this additional step. 

Please refer to ipython notebook for details about modeling and results. 

I have also written a [post](https://weiminwang.blog/2017/06/23/credit-card-fraud-detection-using-auto-encoder-in-tensorflow-2/) explaining the details. 
