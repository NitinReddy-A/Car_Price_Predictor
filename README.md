# Car Price Prediction

This project demonstrates a car price prediction model using Linear Regression. The dataset used is `CarPrice_Assignment.csv`, which contains various features related to car specifications and prices. The project includes data exploration, visualization, preprocessing, model training, evaluation, and prediction.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
  - [Visualization Techniques](#visualization-techniques)
  - [Feature Importance](#feature-importance)
  - [Price Predictor Function](#price-predictor-function)
  - [Example Usage](#example-usage)


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/car-price-prediction.git
   cd car-price-prediction
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

## Usage

1. Load the dataset:
   ```python
   data = pd.read_csv('CarPrice_Assignment.csv')
2. Follow the steps outlined in the Project Workflow section to preprocess the data, train the model, evaluate its performance, and visualize the results.

## Project Workflow

### Exploratory Data Analysis
Explore the dataset to understand its structure and identify any missing values:

    print(data.head())
    print(data.isnull().sum())
    print(data.describe())
Visualize the distribution of car prices:

    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'], kde=True)
    plt.title('Distribution of Car Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.show()

### Data Preprocessing
Select features and handle missing values:

    features = ['horsepower', 'curbweight', 'enginesize', 'highwaympg']
    X = data[features]
    y = data['price']
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
Scale the features:

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

Split the dataset into training and testing sets:

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

## Model Training

### Initialize and train the Linear Regression model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Model Evaluation

### Evaluate the model's performance using various metrics:

```python
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
```

## Visualization Techniques

### Visualize the actual vs. predicted car prices:

```python
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([0, max(y_test)], [0, max(y_test)], '--r')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Car Prices')
plt.show()
```

## Feature Importance

### Determine the importance of each feature:

```python
importance = model.coef_
importance_df = pd.DataFrame({'Feature': features, 'Coefficient': importance})

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()
```

## Price Predictor Function

### Define a function to predict car prices based on input features:

```python
def predict_price(horsepower, curbweight, enginesize, highwaympg):
    input_data = pd.DataFrame([[horsepower, curbweight, enginesize, highwaympg]],
                              columns=['horsepower', 'curbweight', 'enginesize', 'highwaympg'])
    input_data_scaled = scaler.transform(input_data)
    predicted_price = model.predict(input_data_scaled)
    return predicted_price[0]
```

## Example Usage

### Predict the price of a car with specified features:

```python
predicted_price = predict_price(horsepower=150, curbweight=3000, enginesize=2.5, highwaympg=25)
print(f'Predicted Car Price: ${predicted_price:.2f}')
```
