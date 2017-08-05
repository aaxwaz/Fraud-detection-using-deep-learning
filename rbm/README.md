# RBM implementation and its application in fraud detection 

The implementation was modified from 
https://gist.github.com/blackecho/db85fab069bd2d6fb3e7

Changes made to the code: 

1. Implemented Momentum for faster convergence.
2. Added in L2 regularisation.
3. Added in methods for retrieving Free Energy as well as Reconstruction Error in validation data.
4. Simplified the code a bit by removing parts of tf summaries (originally not compatible with tf version 1.1 above)
5. Added in a bit utilities such as plotting training loss

Use it in sk-learn style. 

Check rbm_demo.ipynb for fraud detection application demo and results. 
