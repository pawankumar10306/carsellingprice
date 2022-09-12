
from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

car=pd.read_csv('car.csv')

@app.route('/',methods=['GET','POST'])
def home():
   
    name=sorted(car['name'].unique())
    fuel=sorted(car['fuel'].unique())
    seller=sorted(car['seller_type'].unique())
    gear=sorted(car['transmission'].unique())
    owner=sorted(car['owner'].unique())
    return render_template('index.html',name=name,fuel=fuel,seller=seller,gear=gear,owner=owner)

@app.route('/predict',methods=['POST'])
def predict():
    
    model=pickle.load(open('car.pkl','rb'))
    name=request.form.get('name')
    year=request.form.get('year')
    km=request.form.get('km')
    fuel=request.form.get('fuel')
    seller=request.form.get('seller')
    gear=request.form.get('gear')
    owner=request.form.get('owner')

   
    input_query = pd.DataFrame({'name':[name],'year':[year],'km_driven':[km],
        'fuel':[fuel],'seller_type':[seller],'transmission':[gear],'owner':[owner]})

    result = model.predict(input_query)
    return str(round(result[0],2))

if __name__ == '__main__':
    app.run(debug=True)