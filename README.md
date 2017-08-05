# Credit Card Fraud Detection

Original posts: 
1. Fraud Detection using Auto-encoder: https://weiminwang.blog/2017/06/23/credit-card-fraud-detection-using-auto-encoder-in-tensorflow-2/
2. Fraud Detection using RBM: https://weiminwang.blog/2017/08/05/credit-card-fraud-detection-2-using-restricted-boltzmann-machine-in-tensorflow/

# Data set description 

The data set is available on Kaggle for download - https://www.kaggle.com/dalpozz/creditcardfraud

In summary, it contains 284,807 credit card transactions over 48 hours. It totally contains two types of transactons - fraud and genuine - inside the label column named 'Class'

It is pretty unbalanced - around 0.17% are fraud and the rest are all genuine. The challenge is to build a decent model that is able to tell which are fraud transactions based on the given features of around 30 in total. 

# Models used 

RBM as well as Auto-encoder

Both have achieved around 0.95 AUC score. 