# Federated Traffic Prediction

## Project Overview
This project is a research prototype for privacy-aware network traffic forecasting using Federated Learning.

It uses the GÉANT traffic matrix dataset to explore how multiple simulated clients can collaboratively train a shared traffic forecasting model without pooling their original traffic examples.

## Features
- Interactive network traffic dataset explorer
- Historical traffic visualization
- Four-client federated learning simulation
- Federated Averaging (FedAvg) training demonstration
- Model convergence visualization
- Forecasting model comparison and evaluation metrics

## Technologies Used
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn

## Running the Application

1. Install Python.
2. Install the required dependencies.
3. Run the application using:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Important Note
This application is a research demonstration. The interactive workflow visualization is illustrative, while the live training experiment is a lightweight implementation and should not be interpreted as an exact reproduction of every experiment in the research paper.

## Dataset
The project uses the GÉANT traffic matrix dataset. Refer to the original dataset repository and its licensing terms before redistributing the data.

## Authors
Student research project — VIT Vellore.
