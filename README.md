# Customer Conversion Prediction for a New Age Insurance Company

![Insurance](https://your-image-url.com)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Output Variable](#output-variable-desired-target)
- [Minimum Requirements](#minimum-requirements)
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [Usage](#usage)
- [Author](#author)
- [App Deployment](#app-deployment)
- [App Link](#app-link)

## Problem Statement

You are working for a new-age insurance company that employs multiple outreach plans to sell term insurance to customers. Telephonic marketing campaigns remain one of the most effective ways to reach out to people, but they also incur a lot of cost. Therefore, it is important to identify customers who are most likely to convert beforehand so that they can be specifically targeted via calls. The goal of this project is to build a machine learning model that predicts whether a client will subscribe to the insurance.

## Features

1. age (numeric)
2. job: type of job
3. marital: marital status
4. educational_qual: education status
5. call_type: contact communication type
6. day: last contact day of the month (numeric)
7. mon: last contact month of the year
8. dur: last contact duration, in seconds (numeric)
9. num_calls: number of contacts performed during this campaign and for this client
10. prev_outcome: outcome of the previous marketing campaign (categorical: "unknown", "other", "failure", "success")

## Output Variable (Desired Target)

y - has the client subscribed to the insurance?

## Minimum Requirements

It is not sufficient to just fit a model; the model must be analyzed to find the important factors that contribute to the conversion rate. AUROC must be used as a metric to evaluate the performance of the models.

## Introduction

This project was completed as part of the final project for the Master Data Science course by GUVI. The goal was to predict whether a client will subscribe to the insurance based on historical marketing data of the insurance company.

## Data Preprocessing

The data was loaded and preprocessed, including handling missing values and encoding categorical features.

## Exploratory Data Analysis

Data visualization and exploratory data analysis were performed to gain meaningful insights into the dataset.

## Model Building

The initial model used was logistic regression, which achieved a good AUROC score. However, to build a more reliable model considering the domain context, a decent F1 score was required.

## Model Evaluation

To compare and tune the models, Pycaret was used, and feature importances were analyzed.

## Conclusion

In conclusion, a machine learning model was built to predict whether a customer will subscribe to the insurance. The model was evaluated using AUROC and F1 scores to ensure performance and reliability.

## Usage

To use the model, you can either deploy it using the Streamlit app or run the Python script locally.

## Author

This project was developed by [Your Name].

## App Deployment

The machine learning model has been deployed as an interactive web application using Streamlit. You can access the web app by following the link below:

## App Link

[Customer Conversion Prediction App](https://customer-conversion-prediction-for-a-new-age-insurance-company.streamlit.app/)

---
