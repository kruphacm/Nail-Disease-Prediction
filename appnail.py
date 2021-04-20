import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('modelnail.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('HTMLnail.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x)-1 for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = abs(int(prediction[0]))
    res={0:'Nail psoriasis Treatment:Strong Corticosteroids,injection of Corticosteroids and laser treatment',1:'Brittle Splitting Nails    Treatment:Moisturizer',2:'Nail Fungal Infection   Treatment:Antifungal Medication',3:'Onycholysis    Treatment:Treating psoriasis with oral',4:'Ingrown Toenail    Treatment:Surgery',5:'	Onychogryphosis    Treatment:Podiatrist or Dermatologist',6:'Paronychia    Treatment:Tropical or oral antibiotics,corticosteroids'}
    output=res[output]

    return render_template('HTMLnail.html', prediction_text='Predicted Disease is {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
