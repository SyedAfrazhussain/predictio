from flask import Flask, render_template, request, url_for, redirect
import pickle
import pandas as pd


app=Flask(__name__)

model=pickle.load(open('Diabetes.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def predict():
    text1=request.form['1']
    text2=request.form['2']
    text3=request.form['3']
    text4=request.form['4']
    text5=request.form['5']
    text6=request.form['6']
    text7=request.form['7']
    text8=request.form['8']
    
    row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])
    print(row_df)
    prediction=model.predict_proba(row_df)
    output='{0:.{1}}'.format(prediction[0][1],2)

    if output>str(0.5):
        return render_template('index.html',pred='You have chance of having diabetes.\nProbability of having Diabetes is {}'.format(output))
    else:
        return render_template('index.html',pred='You are safe.\n Probability of having diabetes is {}'.format(output))
if __name__=='__main__':
    app.run(debug=True)
