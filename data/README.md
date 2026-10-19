# Dataset setup

Download the IBM Telco Customer Churn CSV from one of these sources:

- https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv
- https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Save the downloaded file exactly as:

`data/raw/Telco-Customer-Churn.csv`

The project expects a column named `Churn`. The raw data folder is ignored by Git so you do not accidentally publish data files.
