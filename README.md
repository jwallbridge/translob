# Code repository for TransLOB 

This is the repository for the paper _Transformers for Limit Order Books_ which uses a CNN for feature extraction followed by a Transformer to predict future price movements from limit order book data. This is research code and some assembly may be required.

The code is in iPython notebooks, of the form translob_vN.ipynb where larger N means later :

* ``translob_v1.ipynb`` uses the FI-2010 environment with CNN + Transformer (run on Colab) 

# The FI-2010 dataset

The FI-2010 dataset is made up of 10 days of 5 stocks from the Helsinki Stock Exchange, operated by Nasdaq Nordic, consisting of 10 orders on each side of the LOB. Event types can be executions, order submissions, and order cancellations and are non-uniform in time. We restrict to normal trading hours (no auction).

There are 149 rows in each file : 40 LOB data points, 104 hand-crafted features and 5 prediction horizons (k=1, 2, 3, 5, 10).
Each column represents an event snapshot. Data is normalized based on prior day mean and standard deviation.

Data for each of 5 stocks is stored consecutively. Let $\alpha = 0.002$. For percentage changes x >= 0.002, we use label 1.0. For percentage change -a < x < a we use label 2.0. For percentage change x <= a we use label 3.0.
