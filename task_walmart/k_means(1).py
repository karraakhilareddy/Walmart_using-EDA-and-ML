import pandas as pd
data = pd.read_csv("D:/DATA_Science_practice/task_walmart/task1.csv")
#------------------------------------------------------------------------------------
from sqlalchemy import create_engine

user = 'root1'
pw ='Reddy2000'
db='univkm_db'

from urllib.parse import quote
engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}")

data.to_sql('walmartkm',con=engine,if_exists='replace',chunksize=1000,index=False)

sql ='select * from  walmartkm;'

df= pd.read_sql_query(sql,engine)

#-----------------------------------------------------------------------------------------
#df.drop(['Unnamed: 0'] , axis = 1 , inplace =True)
#-------------------------------------------------------------------------------------------------------
"""import sweetviz
my_report = sweetviz.analyze([df,'df'])
my_report.show_html("report.html")"""

#-----------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import joblib
num_pipeline = Pipeline([('scale', MinMaxScaler())])
num_pipeline

# Fit the numeric data to the pipeline. Ignoring State column
processed = num_pipeline.fit(df) 

#save the pipeline
joblib.dump(processed,'processed1')


univ_clean = pd.DataFrame(processed.transform(df), columns=df.columns)

univ_clean

#----------------------------------------------------------------------------------------
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
TWSS = []
k = list(range(2, 11))

for i in k:
    kmeans = KMeans(n_clusters = i,n_init=10)
    kmeans.fit(univ_clean)
    TWSS.append(kmeans.inertia_)

TWSS

# ## Creating a scree plot to find out no.of cluster
plt.plot(k, TWSS, 'ro-'); plt.xlabel("No_of_Clusters"); plt.ylabel("total_within_SS")
#----------------------------------------------------------------------------------------
from sklearn import metrics
model = KMeans(n_clusters = 3)
yy = model.fit(univ_clean)

metrics.silhouette_score(univ_clean, model.labels_)
metrics.calinski_harabasz_score(univ_clean, model.labels_)
metrics.davies_bouldin_score(univ_clean, model.labels_)
#--------------------------------------------------------------------------------------
from sklearn.metrics import silhouette_score

silhouette_coefficients = []

for k in range (2,9):
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(univ_clean)
    score = silhouette_score(univ_clean, kmeans.labels_)
    k = k
    Sil_coff = score
    silhouette_coefficients.append([k, Sil_coff])

silhouette_coefficients

sorted(silhouette_coefficients, reverse = True, key = lambda x: x[1])

#---------------------------------------------------------------------------------------
import pickle
pickle.dump(yy, open('Clust_Univ.pkl', 'wb'))

import os
os.getcwd()

# Cluster labels
model.labels_

mb = pd.Series(model.labels_) 
#-----------------------------------------------------------------------------------
df_clust = pd.concat([mb,df], axis = 1)
df_clust = df_clust.rename(columns = {0:'cluster_id'})
df_clust.head()

cluster_agg = df_clust.iloc[:, 1:].groupby(df_clust.cluster_id).mean()
cluster_agg
















