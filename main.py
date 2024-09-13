import pandas as pd

# Load the data
house_data = pd.read_csv("House_for_rent_islamabad_pk.csv")
house_data
#print unique values in each column
for column in house_data.columns:
    print(column)
    print(house_data[column].unique())
    print()
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# Drop unnecessary column
house_data_cleaned = house_data.drop(columns=['Unnamed: 0'])

# Separate features and target
X = house_data_cleaned.drop('Price', axis=1)
y = house_data_cleaned['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Impute missing values for numerical data
numeric_features = ['Area', 'Bedrooms', 'Baths']
numeric_transformer = SimpleImputer(strategy='median')


# Encode categorical data
categorical_features = ['Location']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a modeling pipeline with Gradient Boosting Regressor
gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=0))
])

# Train the Gradient Boosting model
gb_model.fit(X_train, y_train)

# Predict and evaluate the Gradient Boosting model
y_gb_pred = gb_model.predict(X_test)
gb_rmse = mean_squared_error(y_test, y_gb_pred, squared=False)
gb_mae = mean_absolute_error(y_test, y_gb_pred)
gb_r2 = r2_score(y_test, y_gb_pred)

gb_rmse, gb_mae, gb_r2

house_data_cleaned
import joblib

# Save the model to a file
joblib.dump(gb_model, 'house_price_model.pkl')

def predict_price(location, area, bedrooms, baths):
    # Load the trained model from file
    model = joblib.load('house_price_model.pkl')
    
    # Create a data frame for the input
    input_data = pd.DataFrame({
        'Location': [location],
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Baths': [baths]
    })
    
    # Use the model to predict the price
    predicted_price = model.predict(input_data)
    return predicted_price[0]


# Example usage
location = "DHA"  # Example location
area = 11       # Example area in appropriate units
bedrooms = 3      # Example number of bedrooms
baths = 4         # Example number of bathrooms

predicted_price = predict_price(location, area, bedrooms, baths)
print(f"Predicted Price: PKR {predicted_price:.2f}")
