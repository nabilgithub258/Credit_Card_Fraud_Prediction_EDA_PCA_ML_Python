#!/usr/bin/env python
# coding: utf-8

# In[387]:


############################################################################################################################
######################### CREDIT CARD FRAUD DATA SET  ######################################################################
############################################################################################################################


# In[388]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[389]:


df = pd.read_csv('creditcard.csv')


# In[390]:


df.head()                                        #### this is the best data type one can ask for, modelling shouldn't be hard on this one


# In[391]:


df.info()


# In[392]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[393]:


df[df.duplicated()]                           #### lets take care of duplicates now


# In[394]:


df = df.drop_duplicates()


# In[395]:


df[df.duplicated()]                          #### no duplicates left


# In[396]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[397]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[398]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[399]:


df.isnull().any()                   #### no null values found


# In[400]:


######################################################################
############## Part IV - EDA
######################################################################


# In[401]:


df.head()              #### we have 31 cols


# In[402]:


df.Time.value_counts()


# In[403]:


df.Class.value_counts()                      #### although its imbalanced but we wouldn't have much trouble as the data set to train is massive


# In[404]:


df['Amount'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Credit Card Amount Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


#### y axis is in dollars and it seems theres few outliers here, lets see if those outliers are fraud transactions


# In[405]:


df[df.Amount == df.Amount.max()]['Class']         #### interestingly, its not a fraud transaction as we had assumed before


# In[406]:


df[df.Amount > 15000]['Class']                    #### this is more suprising, all the outliers are not fraud transactions


# In[407]:


corr = df.corr()

corr.loc['Class']


# In[408]:


corr.loc['Time']


# In[409]:


fig, ax = plt.subplots(figsize=(30,14))

sns.heatmap(corr,ax=ax,linewidths=0.5,cmap='viridis')

#### V2:0.08, V4:0.12, V8, V11:0.14 are the ones to look out for


# In[410]:


df['V11'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Credit Card V11 Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### interesting


# In[411]:


df.V11.mean()


# In[412]:


df.V11.std()


# In[413]:


custom = {0:'purple',
         1:'red'}

g = sns.jointplot(x=df.V11,y=df.Amount,data=df,hue='Class',palette=custom)

g.fig.set_size_inches(17,9)

#### clearly we see that the amount doesn't really matter much as most of the fraud happens 2.5 V11


# In[414]:


custom = {0:'black',
         1:'green'}

g = sns.jointplot(x=df.V11,y=df.V4,data=df,hue='Class',palette=custom)

g.fig.set_size_inches(17,9)

#### seems like it happens when V11 and V4 are at a particular point, quite intrigued


# In[415]:


from scipy.stats import pearsonr          #### lets see what pearsonr has to say about the relationship


# In[416]:


co_eff,p_value = pearsonr(df.V11,df.Amount)


# In[417]:


co_eff                           #### not looking good


# In[418]:


p_value                          #### obviously not correlated


# In[419]:


co_eff,p_value = pearsonr(df.V11,df.Class)


# In[420]:


co_eff                      #### looking slightly better but obviously not strongly correlated


# In[421]:


p_value                     #### correlated


# In[422]:


df['V4'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Credit Card V4 Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### V4 and V11 are the ones to look out for


# In[423]:


df.V4.mean()


# In[424]:


df.V4.std()


# In[425]:


custom = {0:'black',
         1:'red'}

g = sns.jointplot(x=df.Amount,y=df.V4,data=df,hue='Class',palette=custom)

g.fig.set_size_inches(17,9)

#### seems like V4 higher then 5 is where the fraud lies


# In[426]:


pl = sns.FacetGrid(df,hue='Class',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'V4',fill=True)

pl.set(xlim=(0,df.V4.max()))

pl.add_legend()

#### we can clearly see the curve where fraud starts, its extremely quite revealing


# In[427]:


custom = {0:'green',
          1:'red'}

pl = sns.FacetGrid(df,hue='Class',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'V11',fill=True)

pl.set(xlim=(0,df.V11.max()))

pl.add_legend()


# In[428]:


custom = {0:'purple',
          1:'red'}

pl = sns.FacetGrid(df,hue='Class',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'V2',fill=True)

pl.set(xlim=(0,df.V2.max()))

pl.add_legend()

#### from this it seems the model will have a easy time predicting, good for us


# In[429]:


custom = {0:'black',
          1:'red'}

sns.lmplot(x='V11',y='V2',data=df,height=7,aspect=2,hue='Class',palette=custom)

#### with fraud transaction we can clearly see some linear relationship 


# In[430]:


sns.lmplot(x='V11',y='Class',data=df,height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'},x_bins=[-5,-4.5,-4,-3,-2,-1,0,1.5,3,6,6.5,6.9,7.5,9,10,12])

#### clearly we see as the V11 passes 5.0 threshold the fraud starts to appear


# In[431]:


sns.lmplot(x='V4',y='Class',data=df,height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'},x_bins=([-5,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,15]))

plt.savefig('Credit_V4_Class_lmplot.jpeg', dpi=300, bbox_inches='tight')

#### similar case here, modelling will have a field day with such data, this is perfect data to model on


# In[432]:


######################################################################
############## Part V - PCA
######################################################################


# In[433]:


X = df.drop(columns='Class')

X.head()


# In[434]:


y = df['Class']

y.head()


# In[435]:


from sklearn.preprocessing import StandardScaler

#### a data set like this is most ideal for PCA


# In[436]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[437]:


from sklearn.decomposition import PCA


# In[438]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
final_df = pd.concat([principal_df, y], axis=1)


# In[439]:


final_df.head()


# In[440]:


final_df.isnull().any()


# In[441]:


final_df[final_df.principal_component_1.isna()]


# In[442]:


final_df.info()


# In[443]:


final_df[final_df.principal_component_2.isna()]


# In[444]:


final_df[final_df.Class.isna()]


# In[445]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(final_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[446]:


final_df = final_df.dropna()


# In[447]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(final_df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[448]:


final_df.isnull().any()


# In[449]:


final_df.info()


# In[450]:


colors = {0: 'green', 1: 'red'}

plt.figure(figsize=(15, 6))

for i in final_df['Class'].unique():
    subset = final_df[final_df['Class'] == i]
    plt.scatter(subset['principal_component_1'], subset['principal_component_2'], 
                color=colors[i], label=f'Class = {i}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Titanic Dataset')
plt.legend()
plt.grid(True)

#### beauty of PCA


# In[451]:


features = X.columns

features


# In[452]:


df_comp = pd.DataFrame(pca.components_,columns=[features])


# In[453]:


df_comp


# In[454]:


fig, ax = plt.subplots(figsize=(20,8))                     

sns.heatmap(df_comp,ax=ax,linewidths=0.5,annot=True,cmap='viridis')

#### PCA corr heatmap


# In[ ]:


#######################################################################
######################## Part VI - Model - Classification
#######################################################################


# In[189]:


from statsmodels.tools.tools import add_constant

df_with_constant = add_constant(df)

df_with_constant.head()                    #### setting up Vif


# In[190]:


vif = pd.DataFrame()                      #### this is extremely helpful and important to know which col can be a problem


# In[191]:


vif["Feature"] = df_with_constant.columns


# In[192]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif["VIF"] = [variance_inflation_factor(df_with_constant.values, i) for i in range(df_with_constant.shape[1])]


# In[193]:


vif                 #### amount will be a problem if you follow the book but we will tackle another way


# In[194]:


from sklearn.model_selection import train_test_split


# In[195]:


X = df.drop(columns=['Class'])

X.head()


# In[196]:


y = df['Class']

y.value_counts()                       #### just look at the imbalance but thankfully our data is massive so it shouldnt throw off our model


# In[197]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# In[198]:


X.columns


# In[199]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Amount'])
    ])


# In[200]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[201]:


from sklearn.linear_model import LogisticRegression


# In[202]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])


# In[203]:


model.fit(X_train,y_train)


# In[204]:


y_predict = model.predict(X_test)


# In[205]:


from sklearn import metrics


# In[206]:


print(metrics.classification_report(y_test,y_predict))                #### quite decent model if you ask me


# In[207]:


from sklearn.ensemble import RandomForestClassifier


# In[208]:


from sklearn.ensemble import RandomForestClassifier

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])


# In[209]:


model.fit(X_train,y_train)


# In[210]:


y_predict = model.predict(X_test)


# In[211]:


print(metrics.classification_report(y_test,y_predict))                   #### this is a very good result honestly


# In[212]:


from sklearn.linear_model import RidgeClassifier                     #### lets see what ridge can bring to the table


# In[213]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Amount'])
    ])


# In[214]:


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RidgeClassifier(alpha=1.0))
])


# In[215]:


model.fit(X_train,y_train)


# In[216]:


y_predict = model.predict(X_test)


# In[217]:


print(metrics.classification_report(y_test,y_predict))                     #### didn't help much


# In[218]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components='mle'))  
        ]),['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28','Amount'])
    ])


# In[223]:


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


# In[224]:


import xgboost as xgb


# In[225]:


clf_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',n_jobs=-1))
])

param_grid_xgb = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.7, 0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.8, 0.9]
}


# In[226]:


from sklearn.model_selection import RandomizedSearchCV


# In[228]:


get_ipython().run_cell_magic('time', '', "\nrandom_search_xgb = RandomizedSearchCV(clf_xgb, param_grid_xgb, cv=3, scoring='accuracy', random_state=42,verbose=2)\nrandom_search_xgb.fit(X_train, y_train)")


# In[230]:


best_model = random_search_xgb.best_estimator_


# In[231]:


y_predict = best_model.predict(X_test)


# In[232]:


print(metrics.classification_report(y_test,y_predict))                           #### best one yet


# In[233]:


from sklearn.ensemble import StackingClassifier


# In[239]:


base_models = [
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3,verbose=2)


# In[240]:


get_ipython().run_cell_magic('time', '', "\nmodel = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', stacking_clf)\n])\n\nmodel.fit(X_train, y_train)")


# In[241]:


y_predict = model.predict(X_test)


# In[242]:


print(metrics.classification_report(y_test,y_predict))                           #### no improvement


# In[ ]:


############################################################################################################################
#### We are concluding our model development phase, having achieved outstanding performance in predicting credit card ######
#### fraud. Our model demonstrates near-perfect accuracy of 1.00 and a precision close to 0.95. These results reflect ######
#### the model's exceptional ability to correctly identify fraudulent transactions while minimizing false positives. #######
############################################################################################################################

