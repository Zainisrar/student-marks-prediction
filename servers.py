from flask import Flask, request,render_template
import numpy as np
import joblib

# Load the trained model
model = joblib.load('student_score_prediction.pkl')

# Create a Flask app
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
# Define a route for the prediction API

@app.route('/predict',methods=['GET','POST'])
def predict():

    # prediction
    global df
    
    input_features = [int(x) for x in request.form.values()] #list comparision
    features_value = np.array(input_features)
    print(features_value)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >12:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 12 if you live on the Earth')
      

    output = model.predict([features_value])[0][0].round(2)


    return render_template('index.html', prediction_text='You will get {} Percentage marks Dear Student, when you do study {} hours per day'.format(output*100, int(features_value[0])))

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
