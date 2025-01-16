# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:15:57 2025

@author: User
"""

import pandas as pd
import numpy as np



df_imbd_box_offiice = pd.read_csv(r"C:\Users\User\.spyder-py3\Poco\imbd_marvel.csv")
df_superhero_movie= pd.read_csv(r"C:\Users\User\.spyder-py3\Poco\superhero-movie-dataset-1978-2012-header.csv")
df_superhero_movie['Index'] = np.arange(0, df_superhero_movie['Title'].shape[0])

df_superhero_movie['Opening Weekend Attendance'] = df_superhero_movie['Opening Weekend Attendance'].fillna(0)

rounded_data_weeknd_attend =  df_superhero_movie['Opening Weekend Attendance'].fillna(0).round(0).astype('int64')

poulation_rate_attendd_movie = rounded_data_weeknd_attend/df_superhero_movie['US Population That Year']*100

df_superhero_movie['Population Rate attend the Movie'] = poulation_rate_attendd_movie

composite_score_prop = df_superhero_movie['IMDB Score']*10*0.7 +  df_superhero_movie['RT Score']*0.3

df_superhero_movie['Senay Composite Score '] = composite_score_prop


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(df_superhero_movie[['Opening Weekend  Box Office']])
scaling_dataset_x = scaler.transform(df_superhero_movie[['Opening Weekend  Box Office']])*100
scaling_weekend_attendce = pd.Series(scaling_dataset_x.reshape(49,))

df_conclusion = df_superhero_movie['Senay Composite Score ']*0.7 + df_superhero_movie ['Population Rate attend the Movie']*0.2 + scaling_weekend_attendce*0.1


df_superhero_movie['Result'] = df_conclusion

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.title('Performans Movies')

plt.bar(df_superhero_movie['Index'], df_superhero_movie['Result'],color=['red', 'blue', 'green'], width=0.5)

plt.show()



