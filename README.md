# Credit Card Fraud Detection

Original blog posts: 
1. Fraud Detection using Auto-encoder: https://weiminwang.blog/2017/06/23/credit-card-fraud-detection-using-auto-encoder-in-tensorflow-2/
2. Fraud Detection using RBM: https://weiminwang.blog/2017/08/05/credit-card-fraud-detection-2-using-restricted-boltzmann-machine-in-tensorflow/

# Data set description 

The data set is available on Kaggle for download - https://www.kaggle.com/dalpozz/creditcardfraud

In summary, it contains 284,807 credit card transactions over 48 hours. It totally contains two types of transactons - fraud and genuine - inside the label column named 'Class'

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, no more information is provided regarding the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise."

It is pretty unbalanced - around 0.17% are fraud and the rest are all genuine. The challenge is to build a decent model that is able to tell which are fraud transactions based on the given features of around 30 in total. Area Under the ROC Curve (AUC) score is recommended as model evaluation criterion given the unbalanced nature of data. 

# Models used in this tutorial 

RBM as well as Auto-encoder

Both have achieved around 0.95 AUC score on val set. 

