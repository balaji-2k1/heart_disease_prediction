
from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

# @app.route('/classify.html')
# def classify():
#     return render_template('classification.html')

@app.route('/predict.html',methods=['GET','POST'])
def get_vals():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        chest_pain_type = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fasting_bp= float(request.form['fbs'])
        rest_ecg = float(request.form['restecg'])
        max_heartrate = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        major_vessels = float(request.form['ca'])
        thal = float(request.form['thal'])

        args=[age,sex,chest_pain_type,trestbps,chol,fasting_bp,rest_ecg,max_heartrate,exang,oldpeak,slope,major_vessels,thal]
        # load_scale=open('/home/balajih2k1/mysite/standard_scaler.pkl','rb')
        scale_data = pickle.load(open('/home/balajih2k1/mysite/standard_scaler.pkl','rb'))

        temp_data = scale_data.transform([args])
        ml_model = pickle.load(open('/home/balajih2k1/mysite/forest_classifier.pkl','rb'))
        # ml_model=joblib.load(temp)
        pred_val=ml_model.predict(temp_data)

        if pred_val==1:
            result='Affected'
        else:
            result='Not Affected'

    return render_template('predict.html',prediction=result)

