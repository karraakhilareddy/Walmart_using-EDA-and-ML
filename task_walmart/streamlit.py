# Import libraries
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import pickle, joblib

processed1 = joblib.load('D:/DATA_Science_practice/task_walmart/processed1')  # Imputation and Scaling pipeline
model = pickle.load(open('D:/DATA_Science_practice/task_walmart/Clust_Univ.pkl', 'rb')) 

def predict(data, user, pw, db,username):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    data1 = pd.DataFrame(processed1.transform(data), columns = data.columns)
    prediction = pd.DataFrame(model.predict(data1), columns = ['cluster_id'])
    prediction1 = pd.concat([prediction, data], axis = 1)
    
    prediction1.to_sql('university_pred_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
    welcome_message =  f'welcome {username}'
    html_table = prediction1.to_html(classes = 'table table-striped')
   

    return prediction1 

def main():
    st.title("Walmart")
    

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Walmart </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
   
    uploadedFile = st.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:light_green;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    user = st.text_input("user", "Type Here")
    pw = st.text_input("password", "Type Here")
    db = st.text_input("database", "Type Here")
    username=st.text_input("enter user_name","Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict(data, user, pw, db,username)
                                   
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm))
    
    
if __name__=='__main__':
    main()