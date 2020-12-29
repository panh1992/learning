import traceback
import sys

import pandas as pd
from flask import request
from flask import Flask
from flask import jsonify
from joblib import load

app = Flask(__name__)


# Your API endpoint URL would consist /predict
@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns = model_columns, fill_value=0)
            prediction = list(lr.predict(query))
            return jsonify({'prediction': str(prediction)})
        except Exception:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print('Train the model first')
        return 'No model here to use'


if __name__ == '__main__':

    try:
        port = int(sys.argv[1])
    except Exception:
        port = 9000
    # Load  model.joblib
    lr = load('model.joblib')
    print('Model loaded')
    # Load model_columns.joblib
    model_columns = load('model_columns.joblib')
    print('Model columns loaded')
    app.run(host='localhost', port=port, debug=True)
