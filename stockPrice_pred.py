#Stock Price Prediction using Historical data with Linear regression
#Dataset contains dates and closing prices

import pandas as pd #for data manipulation and handling time-series data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #to pred stock prices
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Sample historical stock price data
data= {
    'Date': pd.date_range(start='2024-01-01', periods=10, freq='D'), #generates a date range for 10 consecutive days starting from start date
    'Close': [150, 152, 153, 155, 154, 156, 157, 158, 159, 160]
}

#convert dataset into DataFrame
df= pd.DataFrame(data)

#convert date column to numerical(ordinal) format ->represents # of days since a fixed point in history. Transformation needed since lin reg model only works with numerical input
df['Date'] = df['Date'].apply(lambda d: d.toordinal())

print("Stock Price Data:")
print(df.head())

X= df[['Date']]
y= df['Close']

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"\nR2 Score: {r2}")

plt.figure(figsize=(10,6))
plt.plot(X_test, y_test, label="Actual Stock Prices", marker='o')
plt.plot(X_test, y_pred, label="Predicted Stock Prices", marker='x')
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Date (Ordinal)")
plt.ylabel("Stock Price ($)")
plt.legend()
plt.show()

#Test model with a new dataset/future date
future_date = pd.Timestamp('2024-01-11').toordinal()
future_date = pd.DataFrame({'Date': [future_date]})

predicted_price = model.predict(future_date)
print(f"Predicted stock price: ${predicted_price[0]:.2f}")