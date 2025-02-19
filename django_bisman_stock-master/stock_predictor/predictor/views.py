import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import io
import base64
import os
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def fetch_stock_data(stock="GOOG"):
    """ Fetch stock data from Yahoo Finance. """
    from datetime import datetime
    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    return yf.download(stock, start, end)


def plot_to_base64(fig):
    """ Convert Matplotlib figure to a base64 string. """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{image_base64}"


def stock_plot(request):
    

    """ Django view to fetch stock data, generate a plot, and display it. """
    stock = request.GET.get("stock", "GOOG")
    google_data = fetch_stock_data(stock)
    # Get absolute path of the model
    
    MODEL_PATH = os.path.join(settings.BASE_DIR, "predictor/static/models/Latest_stock_price_model.keras")

    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found at", MODEL_PATH)
    else:
        print("Model file found at", MODEL_PATH)
    # Load trained model
    model = load_model(MODEL_PATH)

    # Prepare data
    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data.Close[splitting_len:])
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data, y_data = [], []
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Make predictions
    predictions = model.predict(x_data)
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare data for plotting
    ploting_data = pd.DataFrame(
        {
            'Original Test Data': inv_y_test.reshape(-1),
            'Predictions': inv_pre.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(pd.concat([google_data.Close[:splitting_len + 100], ploting_data], axis=0))
    ax.legend(["Data (Not Used)", "Original Test Data", "Predicted Test Data"])

    # Convert plot to base64
    chart_base64 = plot_to_base64(fig)

    # Pass the chart to the template
    return render(request, "predictor/stock_plot.html", {"chart": chart_base64, "stock": stock})
