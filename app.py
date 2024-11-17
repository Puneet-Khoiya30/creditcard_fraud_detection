from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from a file named 'model.pkl'
with open('model.pkl', 'rb') as f:
    mp = pickle.load(f)

# Route to render the HTML form and handle predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from the form
            values = request.form['values']
            
            # Convert the input string to a list of floats
            input_data = np.array([float(x) for x in values.split(',')]).reshape(1, -1)
            
            # Make a prediction using the loaded model
            prediction = mp.predict(input_data)

            # Check if the prediction is fraudulent or not and return appropriate styled message
            if prediction == 1:
                return render_template('index.html', message="Fraudulent Transaction", message_class="error")
            else:
                return render_template('index.html', message="Non-Fraudulent Transaction", message_class="success")
        except Exception as e:
            # Handle errors (like conversion issues)
            return render_template('index.html', message=f"Error: {str(e)}", message_class="error")
    
    # Render the HTML form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
