# IBMTrader

A TensorFlow-based LSTM model for predicting IBM stock prices using historical data from Alpha Vantage. There are numerous issues with this program as this was the first iteration of this project. As a result, THIS PROJECT IS MEANT FOR EDUCATIONAL/INFORMATIONAL PURPOSES ONLY.

## Features

- Fetches IBM stock data using Alpha Vantage API
- Preprocesses and scales data using `MinMaxScaler`
- Trains a 2-layer LSTM model with dropout and batch normalization
- Predicts closing prices and compares with actual values
- Visualizes predictions using Matplotlib

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- Requests

## Usage

1. Clone the repo:
    ```bash
    git clone https://github.com/yourusername/IBMTrader.git
    cd IBMTrader
    ```

2. (Optional) Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the model:
    ```bash
    python IBMTrader.py
    ```

## Notes

- This script uses the `demo` API key from Alpha Vantage, which is rate-limited.
- `getEndOfYearIndex()` is the seperation date between training and testing data. The date was changed to 2024-06-03 in the code.
