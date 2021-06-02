import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import Markup
app = Flask(__name__)
model = pickle.load(open('modelnail.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLnail.html')

@app.route('/predict',methods=['POST'])
def predict():
    res1={'Crumbling Nail':1.0,'Pitting':2.0,'Change in color,Blood under the nails':3.0,'The nail separates from the bed':4.0,'Nail breaks easily':5.0,"Affects both finger nail and toe nail":6.0,'Drying the nails':7.0,'Typically affects only finger nail':8.0,'Thick Nail':9.0,'Discolored nail that are brown,yellow,white':10.0,'Fragile and cracked nail':11.0,'Discoloration of nail yellow,green or opaque':12.0,'Nail pitting,Nail thickening':13.0,'Bending of nail edges':14.0,'Swelling,Tenderness':15.0,'Redness,soreness':16.0,'Pus':17.0,'Genetics,Injury':18.0,'Circulation issues':19.0,'Ichthyosis':20.0,'Swelling':21.0,'Pain,redness':22.0,'Fever and gland pain':23.0,'Yellow pus':24.0}
    int_features = [res1[x] for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
    print(final_features)
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'<p>Predicted Disease is Nail psoriasis<br><br>Treatment:Strong Corticosteroids,injection of Corticosteroids and laser treatment</p>',1:'<p>Predicted Disease is Brittle Splitting Nails<br><br>Treatment:Moisturizer</p>',2:'<p>Predicted Disease is Nail Fungal Infection<br><br>Treatment:Antifungal Medication</p>',3:'<p>Predicted Disease is Onycholysis<br><br>Treatment:Treating psoriasis with oral</p>',4:'<p>Predicted Disease is Ingrown Toenail<br><br>Treatment:Surgery</p>',5:'<p>Predicted Disease is Onychogryphosis<br><br>Treatment:Podiatrist or Dermatologist</p>',6:'<p>Predicted Disease is Paronychia<br><br>Treatment:Tropical or oral antibiotics,corticosteroids</p>'}
    output=res[output]
    output=Markup(output)
    return render_template('HTMLnail.html', prediction_text=output)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
