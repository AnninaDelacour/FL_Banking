# Federated Learning in Banking
This repository contains the implementation of a Federated Learning (FL) system for financial institutions using XGBoost. 
The project focuses on financial health assessment across multiple institutions using Federated Learning techniques.


### Project structure

FLMODEL  
│── FL_XGBoost/                  # Core implementation of Federated Learning  
│   │── __init__.py               # Package initialization  
│   │── client_app.py             # Client-side implementation for FL training  
│   │── data_partition.py         # Data preprocessing and partitioning for FL  
│   │── global_model.py           # Centralized model aggregation  
│   │── server_app.py             # Server-side orchestration of FL  
│  
│── jupyter_notebooks/            # Jupyter notebooks for analysis and visualization  
│── .gitignore                     # Files to be ignored by Git  
│── BankData.zip                    # Zipped dataset files  
│── Feature_Engineering.ipynb        # Feature selection and preprocessing  
│── pyproject.toml                    # Dependencies and environment configuration  
│── requirements.txt                   # Python dependencies  
│── top10_feature_importance.png       # Feature importance visualization  
