from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["*"])  # Configure CORS properly

# Global variables for model and data
model = None
car_data = None

def load_resources():
    """Load model and data with proper error handling"""
    global model, car_data
    
    try:
        # Load model
        if os.path.exists('LinearRegressionModel.pkl'):
            with open('LinearRegressionModel.pkl', 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        else:
            logger.error("LinearRegressionModel.pkl not found")
            return False
            
        # Load data
        if os.path.exists('cleaned_car.csv'):
            car_data = pd.read_csv('cleaned_car.csv')
            logger.info(f"Data loaded successfully - {len(car_data)} records")
        else:
            logger.error("cleaned_car.csv not found")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        return False

def validate_input_data(company, car_model, year, fuel_type, driven):
    """Validate user input data"""
    errors = []
    
    # Check required fields
    if not all([company, car_model, year, fuel_type, driven]):
        errors.append("All fields are required")
        
    # Check if company is selected
    if company == 'Select Company' or not company:
        errors.append("Please select a valid company")
        
    # Validate year
    try:
        year_int = int(year)
        current_year = datetime.now().year
        if year_int < 1900 or year_int > current_year:
            errors.append(f"Year must be between 1900 and {current_year}")
    except (ValueError, TypeError):
        errors.append("Year must be a valid number")
        
    # Validate kilometers driven
    try:
        driven_float = float(driven)
        if driven_float < 0:
            errors.append("Kilometers driven cannot be negative")
        elif driven_float > 1000000:
            errors.append("Kilometers driven value seems unrealistic")
    except (ValueError, TypeError):
        errors.append("Kilometers driven must be a valid number")
        
    return errors

@app.route("/", methods=['GET'])
def index():
    """Main page route"""
    try:
        if not car_data is not None:
            return jsonify({"error": "Car data not available"}), 500
            
        # Prepare data for template
        companies = sorted(car_data['company'].unique().tolist())
        car_models = sorted(car_data['name'].unique().tolist())
        years = sorted(car_data['year'].unique().tolist(), reverse=True)
        fuel_types = sorted(car_data['fuel_type'].unique().tolist())
        
        # Add default option for companies
        companies.insert(0, 'Select Company')
        
        return render_template(
            'index.html',
            companies=companies,
            car_models=car_models,
            years=years,
            fuel_types=fuel_types
        )
        
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route"""
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 500
            
        # Get form data
        company = request.form.get('company', '').strip()
        car_model = request.form.get('car_model', '').strip()  # Fixed: was car_models
        year = request.form.get('year', '').strip()
        fuel_type = request.form.get('fuel_type', '').strip()
        driven = request.form.get('kilometers', '').strip()  # Fixed: was kilo_driven
        
        logger.info(f"Prediction request: {company}, {car_model}, {year}, {fuel_type}, {driven}")
        
        # Validate input
        validation_errors = validate_input_data(company, car_model, year, fuel_type, driven)
        if validation_errors:
            return jsonify({"error": "; ".join(validation_errors)}), 400
            
        # Convert data types
        year_int = int(year)
        driven_float = float(driven)
        
        # Create prediction dataframe with exact column names expected by model
        prediction_data = pd.DataFrame({
            'name': [car_model],
            'company': [company],
            'year': [year_int],
            'kms_driven': [driven_float],
            'fuel_type': [fuel_type]
        })
        
        # Make prediction
        prediction = model.predict(prediction_data)
        result = float(np.round(prediction[0], 2))
        
        logger.info(f"Prediction successful: {result}")
        
        # Return as JSON for better handling
        return jsonify({
            "prediction": result,
            "formatted": f"₹{result:,.2f}"
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500

@app.route('/api/car-models/<company>')
def get_car_models(company):
    """API endpoint to get car models for a specific company"""
    try:
        if car_data is None:
            return jsonify({"error": "Data not available"}), 500
            
        # Filter car models by company
        company_models = car_data[car_data['company'] == company]['name'].unique().tolist()
        company_models.sort()
        
        return jsonify({"models": company_models})
        
    except Exception as e:
        logger.error(f"Error getting car models: {str(e)}")
        return jsonify({"error": "Failed to load car models"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if (model is not None and car_data is not None) else "unhealthy",
        "model_loaded": model is not None,
        "data_loaded": car_data is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Page not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({"error": "Internal server error"}), 500

# Initialize resources on startup
if not load_resources():
    logger.error("Failed to load required resources. Application may not work properly.")

if __name__ == '__main__':
    app.run(
        debug=False,  # Set to False for production
        host='0.0.0.0',
        port=5000,
        threaded=True
    )