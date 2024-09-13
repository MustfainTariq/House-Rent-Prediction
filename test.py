import unittest
from unittest.mock import patch
import pandas as pd
import joblib
from pandas.testing import assert_frame_equal

# Assuming the predict_price function is defined in app.py
from app import predict_price

class TestPredictPrice(unittest.TestCase):

    @patch('app.joblib.load')
    def test_predict_price(self, mock_joblib_load):
        # Mock the model's predict method
        mock_model = unittest.mock.MagicMock()
        mock_model.predict.return_value = [20000000]  # Example predicted price
        mock_joblib_load.return_value = mock_model
        
        # Example input values
        location = "DHA"
        area = 11
        bedrooms = 3
        baths = 4
        
        # Call the predict_price function
        predicted_price = predict_price(location, area, bedrooms, baths)
        
        # Assert that the predicted price is correct
        self.assertEqual(predicted_price, 20000000)

        # Check that the model's predict method was called with the correct input
        expected_input_data = pd.DataFrame({
            'Location': [location],
            'Area': [area],
            'Bedrooms': [bedrooms],
            'Baths': [baths]
        })
        
        # Get the actual input data passed to predict
        actual_input_data = mock_model.predict.call_args[0][0]
        
        # Use assert_frame_equal to compare the two DataFrames
        assert_frame_equal(actual_input_data, expected_input_data)

if __name__ == '__main__':
    unittest.main()
