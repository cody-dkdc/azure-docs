---
title: How to detect data drift (Preview)
titleSuffix: Azure Machine Learning service
description: Learn how to detect data drift on deployed models in Azure Machine Learning service.
services: machine-learning
ms.service: machine-learning
ms.subservice: core
ms.topic: conceptual
ms.reviewer: jmartens
ms.author: copeters
author: cody-dkdc
ms.date: 06/17/2019
---
Machine learning models are generally only as good as the data they are trained with. Deploying a model to production without monitoring its performance can lead to undetected and detrimental impacts. However, directly calculating model performance may take time, resources, or otherwise be impractical. Instead, we can monitor for data drift. 

We define data drift as *-a change in data that causes degradation in model performance-*. We can compute data drift between the data used to inference/score a model and baseline data, most commonly the data used to train the model. See [Data Drift](./concept-data-drift.md) for details on how data drift is measured. 

> [!Note]
> This service is in (Preview) and limited in configuration options. Please see our [API Documentation](https://docs.microsoft.com/en-us/python/api/azureml-contrib-datadrift/?view=azure-ml-py) and [Release Notes](azure-machine-learning-release-notes) for details and updates. 

# How to detect data drift on deployed models in Azure Machine Learning service (Preview)
With Azure Machine Learning service, you can monitor   the inputs to a model deployed on AKS and compare this data to a baseline dataset – typically, the training dataset for the model. At regular intervals, the inference data is [snapshot and profiled](./how-to-explore-prepare-data.md), then computed against the baseline dataset to produce a drift analysis that: 

* Measures the magnitude of data drift, called the [Drift Coefficient](./concept-data-drift.md)

* Measures the Drift Contribution by Feature, informing which features caused data drift

* Measures the Distance Metrics, currently Wasserstein and Energy Distance are computed 

* Measures the Distributions of Features, currently Kernel Density Estimation

* Send alerts to data drift by email 

In this article, we will show how to train, deploy, and monitor data drift on a model with the Azure ML service. 

## Prerequisites

- If you don’t have an Azure subscription, create a free account before you begin. Try the [free or paid version of Azure Machine Learning service](https://aka.ms/AMLFree) today.

- An Azure Machine Learning service Workspace and the Azure Machine Learning SDK for Python installed. Learn how to get these prerequisites using the [How to configure a development environment](how-to-configure-environment.md) document.

- [Set up your environment](how-to-configure-environment.md) and install the Data Drift SDK, Datasets SDK, and lightgbm package:
    ```python
    aks_config = AksWebservice.deploy_configuration(collect_model_data=True, enable_app_insights=True)
    ``` 
    
    ```
    pip install azureml-contrib-datadrift
    pip install azureml-contrib-datasets
    pip install lightgbm
    ```

## Import Dependencies 
Import dependencies used in this guide:

    ```python
    import json
    import os
    import time
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd
    import requests
    
    # Azure ML service packages 
    from azureml.contrib.datadrift import DataDriftDetector, AlertConfiguration
    from azureml.contrib.opendatasets import NoaaIsdWeather
    from azureml.core import Dataset, Workspace, Run
    from azureml.core.compute import AksCompute, ComputeTarget
    from azureml.core.conda_dependencies import CondaDependencies
    from azureml.core.experiment import Experiment
    from azureml.core.image import ContainerImage
    from azureml.core.model import Model
    from azureml.core.webservice import Webservice, AksWebservice
    from azureml.widgets import RunDetails
    from sklearn.externals import joblib
    from sklearn.model_selection import train_test_split    ``` 
    
## Set up 

Set up naming for this guide:

  ```python 
  # set a prefix
  prefix = 'dkdc'
  
  # model, image, and service name to be used in the Azure Machine Learning service
  model_name = '{}DriftModel'.format(prefix)
  image_name = '{}DriftImage'.format(prefix)
  service_name = '{}DriftDeployment'.format(prefix)
  dataset_name = '{}DriftTrainingDataset'.format(prefix)
  
  # set email address that will recieve the alert of data drift
  email_address = ''
  ```

## Get Azure Machine Learning service Workspace
Learn how to create this with the [How to configure a development environment](how-to-configure-environment.md) document.

    ```python
    # load workspace object from config file
    ws = Workspace.from_config()
    
    # print workspace details
    print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
    ```

## Generate Training and Test Data
For this guide, we will use NOAA ISD weather data from [Azure Open Datasets](https://azure.microsoft.com/en-us/services/open-datasets/) to demonstrate the Data Drift service.

First, create functions for getting and formatting the data.

    ```python
    usaf_list = ['725724', '722149', '723090', '722159', '723910', '720279',
             '725513', '725254', '726430', '720381', '723074', '726682',
             '725486', '727883', '723177', '722075', '723086', '724053',
             '725070', '722073', '726060', '725224', '725260', '724520',
             '720305', '724020', '726510', '725126', '722523', '703333',
             '722249', '722728', '725483', '722972', '724975', '742079',
             '727468', '722193', '725624', '722030', '726380', '720309',
             '722071', '720326', '725415', '724504', '725665', '725424',
             '725066']

    columns = ['usaf', 'wban', 'datetime', 'latitude', 'longitude', 'elevation', 'windAngle', 'windSpeed', 'temperature', 'stationName', 'p_k']


    def enrich_weather_noaa_data(noaa_df):
        hours_in_day = 23
        week_in_year = 52

        noaa_df["hour"] = noaa_df["datetime"].dt.hour
        noaa_df["weekofyear"] = noaa_df["datetime"].dt.week

        noaa_df["sine_weekofyear"] = noaa_df['datetime'].transform(lambda x: np.sin((2*np.pi*x.dt.week-1)/week_in_year))
        noaa_df["cosine_weekofyear"] = noaa_df['datetime'].transform(lambda x: np.cos((2*np.pi*x.dt.week-1)/week_in_year))

        noaa_df["sine_hourofday"] = noaa_df['datetime'].transform(lambda x: np.sin(2*np.pi*x.dt.hour/hours_in_day))
        noaa_df["cosine_hourofday"] = noaa_df['datetime'].transform(lambda x: np.cos(2*np.pi*x.dt.hour/hours_in_day))

        return noaa_df

    def add_window_col(input_df):
        shift_interval = pd.Timedelta('-7 days') # your X days interval
        df_shifted = input_df.copy()
        df_shifted['datetime'] = df_shifted['datetime'] - shift_interval
        df_shifted.drop(list(input_df.columns.difference(['datetime', 'usaf', 'wban', 'sine_hourofday', 'temperature'])), axis=1, inplace=True)

        # merge, keeping only observations where -1 lag is present
        df2 = pd.merge(input_df,
                       df_shifted,
                       on=['datetime', 'usaf', 'wban', 'sine_hourofday'],
                       how='inner',  # use 'left' to keep observations without lags
                       suffixes=['', '-7'])
        return df2

    def get_noaa_data(start_time, end_time, cols, station_list):
        isd = NoaaIsdWeather(start_time, end_time, cols=cols)
        # Read into Pandas data frame.
        noaa_df = isd.to_pandas_dataframe()
        noaa_df = noaa_df.rename(columns={"stationName": "station_name"})

        df_filtered = noaa_df[noaa_df["usaf"].isin(station_list)]
        df_filtered.reset_index(drop=True)

        # Enrich with time features
        df_enriched = enrich_weather_noaa_data(df_filtered)

        return df_enriched

    def get_featurized_noaa_df(start_time, end_time, cols, station_list):
        df_1 = get_noaa_data(start_time - timedelta(days=7), start_time - timedelta(seconds=1), cols, station_list)
        df_2 = get_noaa_data(start_time, end_time, cols, station_list)
        noaa_df = pd.concat([df_1, df_2])

        print("Adding window feature")
        df_window = add_window_col(noaa_df)

        cat_columns = df_window.dtypes == object
        cat_columns = cat_columns[cat_columns == True]

        print("Encoding categorical columns")
        df_encoded = pd.get_dummies(df_window, columns=cat_columns.keys().tolist())
    
        print("Dropping unnecessary columns")
        df_featurized = df_encoded.drop(['windAngle', 'windSpeed', 'datetime', 'elevation'], axis=1).dropna().drop_duplicates()
        
        return df_featurized
    ```
    
For this guide, we will try to predict the temperature for each day of the next week, given the data from the past 2 weeks. Naively, we will only train on 2 weeks of data from January 2009.

    ```python 
    # load Jan 1-14 2009 data as a dataframe
    df = get_featurized_noaa_df(datetime(2009, 1, 1), datetime(2009, 1, 14, 23, 59, 59), columns, usaf_list)
    df.head()
    ```

We will split the data for training the model and write the data to a CSV for uploading and registration as an Azure Machine Learning dataset. This training dataset will be associated with the model and used as the baseline dataset for the drift service. 

    ```python
    # generate X and y dataframes for training and testing the model
    label = "temperature"
    x_df = df.drop(label, axis=1)
    y_df = df[[label]]
    x_train, x_test, y_train, y_test = train_test_split(df, y_df, test_size=0.2, random_state=223)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # create directory to write training data to
    training_dir = 'outputs/training'
    training_file = "training.csv"

    # write dataframe to csv to we can upload and register it as a dataset
    os.makedirs(training_dir, exist_ok=True)
    training_df = pd.merge(x_train.drop(label, axis=1), y_train, left_index=True, right_index=True)
    training_df.to_csv(training_dir + "/" + training_file)
    ```
    
## Upload, Create, Register, and Snapshot the Training Dataset

For the Data Drift Service (Preview) uses a baseline dataset for comparison against inference data. In this step we will create the training dataset which is automatically used as the baseline dataset by the service.


    ```python
    
 
    name_suffix = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S")
    snapshot_name = "snapshot-{}".format(name_suffix)

    dstore = ws.get_default_datastore()
    dstore.upload(training_dir, "data/training", show_progress=True)
    dpath = dstore.path("data/training/training.csv")
    trainingDataset = Dataset.auto_read_files(dpath, include_path=True)
    trainingDataset = trainingDataset.register(workspace=ws, name=dataset_name, description="dset", exist_ok=True)

    trainingDataSnapshot = trainingDataset.create_snapshot(snapshot_name=snapshot_name, compute_target=None, create_data_snapshot=True)
    datasets = [(Dataset.Scenario.TRAINING, trainingDataSnapshot)]
    print("dataset registration done.\n")
    datasets
    ```

## Example notebook

The [how-to-use-azureml/deployment/enable-data-collection-for-models-in-aks/enable-data-collection-for-models-in-aks.ipynb](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/deployment/enable-data-collection-for-models-in-aks/enable-data-collection-for-models-in-aks.ipynb) notebook demonstrates concepts in this article.  

[!INCLUDE [aml-clone-in-azure-notebook](../../../includes/aml-clone-for-examples.md)]
