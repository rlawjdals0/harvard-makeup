import flask
from flask import Flask, request, render_template
#from sklearn.externals -->
import joblib
import numpy as np

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def make_prediction():
        if request.method=='POST':

            entered_li = []


            # ---YOUR CODE FOR Part 2.2.3 ----

            # Define "day", "open", "prn", "state", "school", and "store" variables

            day = str(request.form['beds'])
            open = int(request.form['beds'])
            prn = int(request.form['beds'])
            state = int(request.form['beds'])
            school = int(request.form['beds'])
            store = int(request.form['beds'])

            # --- THE END OF CODE FOR Part 2.2.3 ---

            # StoreType, Assortment, CompetitionDistance... arbitrary values
            for val in [1, 1, 290.0, 10.0, 2011.0, 1, 40.0, 2014.0, 0, 0, 0]:
                entered_li.append(val)


                # ---YOUR CODE FOR Part 2.2.4 ----

                # Predict from the model

                RandomForestRegressor(bootstrap=True, ccp_alpha=None, criterion='mse', max_depth=10, max_features='auto', max_lead_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=None, oob_score=False, random_state=10, verbose=0, warm_strat=False)

                rf.fit(X_train, y_train)
                prediction = rf.predict(np.array(val).reshape(1, -1))

                # --- THE END OF CODE FOR Part 2.2.4 ---

                label = str(np.squeeze(prediction.round(2)))

                return render_template('index.html', label=label)


                if __name__ == '__main__':

                    # ---YOUR CODE FOR Part 2.2.1 ----

                    #Load ML model

                    model = joblib.load('./rm.pkl')

                    # --- THE END OF CODE FOR Part 2.2.1 ---

# start API
app.run(host='127.0.0.1', port=5000, debug=True)
