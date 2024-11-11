### DEVELOPED BY: Sri Varshan P
### REGISTER NO: 212222240104
### DATE:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL

## AIM:
To Implement an Auto Regressive Model for Tank loss data using Python
## ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model 
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
## PROGRAM
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error


data = pd.read_csv('/content/russia_losses_equipment.csv', parse_dates=['date'], index_col='date')

print(data.head())

result = adfuller(data['tank'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

model = AutoReg(train['tank'], lags=13)
model_fit = model.fit()

predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test['tank'], predictions)
print('Mean Squared Error:', mse)

plt.figure(figsize=(10,6))
plt.subplot(211)
plot_pacf(train['tank'], lags=13, ax=plt.gca())
plt.title("PACF - Partial Autocorrelation Function")
plt.subplot(212)
plot_acf(train['tank'], lags=13, ax=plt.gca())
plt.title("ACF - Autocorrelation Function")
plt.tight_layout()
plt.show()

print("PREDICTION:")
print(predictions)

plt.figure(figsize=(10,6))
plt.plot(test.index, test['tank'], color='red' ,label='Actual Tank loss')
plt.plot(test.index, predictions, color='purple', label='Predicted Tank loss')
plt.title('Test Data vs Predictions (FINAL PREDICTION)')
plt.xlabel('Date')
plt.ylabel('Tank losses')
plt.legend()
plt.show()

```
## OUTPUT:

### GIVEN DATA

![image](https://github.com/user-attachments/assets/b81ae492-76c1-4417-b405-f4c0e43d4f9b)

### ADF-STATISTIC AND P-VALUE

![image](https://github.com/user-attachments/assets/ae4f8a29-32fb-466d-8f5f-2c67514b729b)


### PACF - ACF

![image](https://github.com/user-attachments/assets/b23da975-b6b8-4be9-8cf9-caf9cb58543e)

### MSE VALUE

![image](https://github.com/user-attachments/assets/d66b1240-a3f6-48d1-b6c8-41ee64365af8)



### PREDICTION

![image](https://github.com/user-attachments/assets/93208a43-75b7-4cd1-b887-fcbdad1000be)

### FINAL PREDICTION

![image](https://github.com/user-attachments/assets/a8fdf366-c06a-42be-a4d4-6237469777ee)


### RESULT:
Thus, the program to implement the auto regression function using python is executed successfully.
