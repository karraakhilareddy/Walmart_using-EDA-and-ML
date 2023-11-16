import pandas as pd
data = pd.read_csv("D:/DATA_Science_practice/Walmart.csv")
#--------------------------------------------------------------------------------
from sqlalchemy  import create_engine

user='root1'
pw='Reddy@2000'
db='univ_db'

from urllib.parse import quote
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote(f'{pw}'))

data.to_sql('univ',con=engine , if_exists='replace',chunksize=1000,index=False)

sql='select * from univ;'

df = pd.read_sql_query(sql,engine)

#--------------------------------------------------------------------------------
import sweetviz
my_report = sweetviz.analyze([df , 'df'])
my_report.show_html('Report.html')

#--------------------------------------------------------------------------------
df.drop(['Company'],axis=1,inplace=True)
#---------------------------------------------------------------------------------
import matplotlib.pyplot as plt
df.plot(kind='box',subplots = True , sharey=False , figsize =(15,8))
plt.subplots_adjust(wspace=0.75)
plt.show()
#----------------------------------------------------------------------------------

from AutoClean import AutoClean
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

clean_line = AutoClean(df , missing_num='auto' ,missing_categ='most_frequent',outliers='winz')
df_cleaned = clean_line.output

#-----------------------------------------------------------------------------------------
categorical = df_cleaned.select_dtypes(include=['object'])
categorical.isna().sum()
numerical = df_cleaned.select_dtypes(include=['number'])
numerical.isna().sum()
#-----------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
label_enc = LabelEncoder()
for column in categorical:
    categorical[column]= label_enc.fit_transform(categorical[column])
#------------------------------------------------------------------------------------------
df_cleaned_final = pd.concat([numerical,categorical],axis=1)
cols = list(df_cleaned_final.columns)
pipe1=make_pipeline(MinMaxScaler())
df_pipelined = pd.DataFrame(pipe1.fit_transform(df_cleaned_final),columns=cols,index = df_cleaned_final.index)

df_pipelined.isna().sum()
#----------------------------------------------------------------------------------------------------------
from scipy.cluster.hierarchy import dendrogram,linkage
plt.figure(1,figsize=(16,8))
tree_plot = dendrogram(linkage(df_pipelined,method='ward'))
plt.title('hierarchial clustering dendogram')
plt.xlabel('index')
plt.ylabel('euclidean distance')
plt.show()
#-------------------------------------------------------------------------------------------------------
from sklearn.cluster import AgglomerativeClustering
model1 = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
model1_output = model1.fit_predict(df_pipelined)
model1.labels_
cluster_labels = pd.Series(model1.labels_)
#-------------------------------------------------------------------------------------------------------
from sklearn import metrics
metrics.silhouette_score(df_pipelined,cluster_labels)
metrics.calinski_harabasz_score(df_pipelined,cluster_labels)
metrics.davies_bouldin_score(df_pipelined,cluster_labels)
#----------------------------------------------------------------------------------------
from clusteval import clusteval
import numpy as np
ce=clusteval(evaluate='silhouette')
df_array=np.array(df_pipelined)
ce.fit(df_array)
ce.plot()
#----------------------------------------------------------------------------------------
model_2=AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean',linkage='complete')
model2_output = model_2.fit_predict(df_pipelined)
model_2.labels_
cluster_label_2 = pd.Series(model_2.labels_)

metrics.silhouette_score(df_pipelined,cluster_label_2)
metrics.calinski_harabasz_score(df_pipelined,cluster_label_2)
metrics.davies_bouldin_score(df_pipelined,cluster_label_2)

#-------------------------------------------------------------------------------------------











