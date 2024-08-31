from datetime import datetime, timedelta
from flask import Flask, render_template, session, url_for, redirect, request, jsonify
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
import joblib
import mysql.connector

# 1. Create an instance of the Flask class
app = Flask(__name__)
app.config['SECRET_KEY'] = 'asecretkey'

# 2. Load the machine learning model
model = joblib.load('sku_reorder_predictor.joblib')

# 3. Database connection function
def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='Vladlale@123',
        database='inventory_db'
    )
    return connection

# 4. Define a prediction function
def check_reorder(prediction, current_stock, avg_daily_sales, lead_time_days, buffer_days=2, use_variability=False):
    if use_variability:
        safety_stock = avg_daily_sales * lead_time_days * buffer_days  # Example using demand variability
    else:
        safety_stock = buffer_days * avg_daily_sales  # Safety stock based on buffer days

    reorder_point = (avg_daily_sales * lead_time_days) + safety_stock
    
    if current_stock < reorder_point:
        reorder_needed = True
        reorder_date = datetime.now() + timedelta(days=prediction)
        days_to_reorder = max(0, int(prediction))
    else:
        reorder_needed = False
        reorder_date = None
        days_to_reorder = None

    return reorder_needed, reorder_point, reorder_date, days_to_reorder

# 5. Create a WTForm Class for SKU input
class PredictForm(FlaskForm):
    avg_daily_sales = FloatField("Average Daily Sales")
    current_stock = FloatField("Current Stock")
    lead_time_days = FloatField("Lead Time (Days)")
    demand_variability = FloatField("Demand Variability")
    submit = SubmitField("Predict Reorder Time")

# 6. Set up the home page
@app.route("/", methods=["GET", "POST"])
def index():
    form = PredictForm()
    if form.validate_on_submit():
        app.logger.info(f"Form submitted with data: {form.data}")
        session['avg_daily_sales'] = form.avg_daily_sales.data
        session['current_stock'] = form.current_stock.data
        session['lead_time_days'] = form.lead_time_days.data
        session['demand_variability'] = form.demand_variability.data
        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)

def return_prediction(model, input_json):
    input_data = [[input_json[k] for k in input_json.keys()]]
    prediction = model.predict(input_data)[0]
    return prediction

# 7. Define a "prediction" route that processes form input and returns a model prediction
@app.route('/prediction', methods=['GET'])
def prediction():
    avg_daily_sales = float(session['avg_daily_sales'])
    current_stock = float(session['current_stock'])
    lead_time_days = float(session['lead_time_days'])
    demand_variability = float(session['demand_variability'])
    
    content = {
        'avg_daily_sales': avg_daily_sales,
        'current_stock': current_stock,
        'lead_time_days': lead_time_days,
        'demand_variability': demand_variability
    }

    predicted_reorder_time = return_prediction(model, content)
    
    is_reorder_needed, reorder_point, reorder_date, days_to_reorder = check_reorder(
        predicted_reorder_time,
        current_stock,
        avg_daily_sales,
        lead_time_days,
        buffer_days=2,  # Buffer days for safety stock
        use_variability=False  # Set to True if you want to use demand variability for safety stock
    )
    response = {
        'is_reorder_needed': is_reorder_needed,
        'reorder_point': reorder_point,
        'reorder_date': reorder_date.strftime('%Y-%m-%d') if reorder_date else None,
        'days_until_reorder': days_to_reorder if is_reorder_needed else None,
        'alert_message': f"Alert: Reorder needed! Current stock is below the reorder point of {reorder_point:.2f} units." if is_reorder_needed else "No reorder needed. Current stock is sufficient."
    }
    
    return jsonify(response)

# 8. Create a GET and POST route for SKU prediction and retrieval
@app.route('/sku/predict', methods=['GET', 'POST'])
def sku_predict():
    if request.method == 'POST':
        # Extract data from the POST request JSON body
        data = request.get_json()
        avg_daily_sales = float(data['avg_daily_sales'])
        current_stock = float(data['current_stock'])
        lead_time_days = float(data['lead_time_days'])
        demand_variability = float(data['demand_variability'])
        
        content = {
            'avg_daily_sales': avg_daily_sales,
            'current_stock': current_stock,
            'lead_time_days': lead_time_days,
            'demand_variability': demand_variability
        }
        
        predicted_reorder_time = return_prediction(model, content)
        
        is_reorder_needed, reorder_point, reorder_date, days_to_reorder = check_reorder(
            predicted_reorder_time,
            current_stock,
            avg_daily_sales,
            lead_time_days,
            buffer_days=2,  # Buffer days for safety stock
            use_variability=False  # Set to True if you want to use demand variability for safety stock
        )
        
        response = {
            'is_reorder_needed': is_reorder_needed,
            'reorder_point': reorder_point,
            'reorder_date': reorder_date.strftime('%Y-%m-%d') if reorder_date else None,
            'days_until_reorder': days_to_reorder if is_reorder_needed else None,
            'alert_message': f"Alert: Reorder needed! Current stock is below the reorder point of {reorder_point:.2f} units." if is_reorder_needed else "No reorder needed. Current stock is sufficient."
        }
        
        return jsonify(response)
    
    # If it's a GET request, return all SKU data
    elif request.method == 'GET':
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM skus")
        skus_data = cursor.fetchall()
        cursor.close()
        connection.close()

        return jsonify(skus_data)

# 9. Define a route to display all SKUs from the database
@app.route('/skus')
def skus():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM skus")
    skus_data = cursor.fetchall()
    cursor.close()
    connection.close()

    return jsonify(skus_data)

# 10. Allows us to run flask using $ python app.py
if __name__ == '__main__':
    app.run(debug=True)
