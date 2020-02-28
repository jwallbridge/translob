# Code repository for TransLOB 

This is the repository for the paper _Transformers for Limit Order Books_ which uses a CNN for feature extraction followed by a Transformer to predict future price movements from limit order book data. 

Paper :  

* ``TransLOB.pdf``.

Python files added shortly :  

* ``LobFeatures.py`` CNN and inception functions.
* ``LobAttention.py`` self-attention function.
* ``LobPosition.py`` positional encodings.
* ``LobTransformer.py`` transformer function.

The code is in iPython notebooks :

* ``translob.ipynb`` uses the FI-2010 environment with CNN + Transformer (run on Colab and GCP). 

This is research code and some assembly may be required.


# FI-2010 dataset

The FI-2010 dataset is made up of 10 days of 5 stocks from the Helsinki Stock Exchange, operated by Nasdaq Nordic, consisting of 10 orders on each side of the LOB. Event types can be executions, order submissions, and order cancellations and are non-uniform in time. We restrict to normal trading hours (no auction).

There are 149 rows in each file : 40 LOB data points, 104 hand-crafted features and 5 prediction horizons (k=10, 20, 30, 50, 100).
Each column represents an event snapshot. Data is normalized based on the prior day mean and standard deviation and is stored consecutively for each of the 5 stocks. 

The training labels for prediction are as follows. Let a = 0.002. For percentage changes x >= 0.002, label 1.0 is used. For percentage changes -a < x < a, label 2.0 is used. For percentage changes x <= -a, label 3.0 is used.
