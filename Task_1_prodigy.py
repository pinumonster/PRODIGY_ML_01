import pandas as pd

# Load the dataset
data = pd.read_csv("HousePrices.csv")

# Display the first few rows of the dataset
data.head()


# Check for missing values
missing_values = data.isnull().sum()

# Drop columns with significant missing values
data_cleaned = data.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'])

# Fill missing values with appropriate statistics, for instance, using median for numerical columns
data_cleaned.fillna(data_cleaned.median(), inplace=True)

# Display the dataset after cleaning
data_cleaned.head()


# Select features
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
X = data_cleaned[features]
y = data_cleaned['Property_Sale_Price']

# Display the selected features and target
X.head(), y.head()


from sklearn.model_selection import train_test_split

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LinearRegression

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")


test_data = pd.read_csv("HousePrices.csv")

# Select the same features as used in training
X_test_new = test_data[features]

# Handle any necessary preprocessing similar to the train set
#X_test_new.fillna(data_cleaned.median(), inplace=True)

# Make predictions
predictions = model.predict(X_test_new)

# Display the predictions
print(predictions)
