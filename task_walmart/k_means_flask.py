# Import libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
import pandas as pd
import pickle
import joblib
processed1 = joblib.load('D:/DATA_Science_practice/task_walmart/processed1')  # Imputation and Scaling pipeline
model = pickle.load(open('D:/DATA_Science_practice/task_walmart/Clust_Univ.pkl', 'rb')) 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['databasename']
        username = request.form['name']
     
        engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
        try:

            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                  
        # Drop the unwanted features
        # data.drop(['Unnamed'] , axis = 1 , inplace =True)
        data1 = pd.DataFrame(processed1.transform(data), columns = data.columns)
        prediction = pd.DataFrame(model.predict(data1), columns = ['cluster_id'])
        prediction = pd.concat([prediction, data], axis = 1)
        
        prediction.to_sql('university_pred_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = prediction.to_html(classes = 'table table-striped')
        welcome_message =  f'welcome {username}'
        return render_template("data.html", welcome_message=welcome_message ,Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")
                
    
    

if __name__=='__main__':
    app.run(debug = True)

