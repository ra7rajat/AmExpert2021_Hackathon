#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from ast import literal_eval


# In[2]:


df = pd.read_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/train_go05W65.csv')
print(df.shape)
print(df.describe())
df.head(2)


# In[3]:


test_df = pd.read_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/test_VkM91FT.csv')
print(test_df.shape)
test_df.head()


# In[4]:


bins= [0, 30, 40, 50, 60, 70]

labels = ['less_than_30','less_than_40','less_than_50','less_than_60','more_than_60']

df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=bins, labels=labels, right=False)

df.head()


# In[5]:


test_df.head()


# In[6]:


df[df['AgeGroup']=='less_than_60']['Age'].describe()


# In[7]:


test_df[test_df['AgeGroup']=='less_than_60']['Age'].describe()


# In[8]:


df['AgeGroup'].value_counts()


# In[9]:


test_df['AgeGroup'].value_counts()


# In[10]:


df.head()


# In[11]:


from sklearn.preprocessing import LabelEncoder

categorical_variables=['Gender', 'City_Category','Customer_Category','AgeGroup']

for i in categorical_variables:
    le = LabelEncoder()
    df[i] = le.fit_transform(df[i])
    test_df[i] = le.fit_transform(test_df[i])
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print("dictionary of variables encoded in column ",i," : \n", mapping)
    
df.head(2)


# In[12]:


test_df.head()


# In[13]:


df['Age'].nunique()


# In[14]:


df['Vintage'].nunique()


# In[15]:


df['Vintage'].isna().sum()


# In[16]:


df.corr()


# In[17]:


df['Age'].describe()


# In[18]:


df['Is_Active'].value_counts()


# In[19]:


df['City_Category'].value_counts()


# In[20]:


df['Customer_Category'].value_counts()


# In[21]:


df.head()


# In[22]:


df['Vintage'].describe()


# In[23]:


65//12


# In[24]:


df['Vintage_year'] = df['Vintage'].apply(lambda x : x//12)
test_df['Vintage_year'] = test_df['Vintage'].apply(lambda x : x//12)
df.head()


# In[25]:


test_df.head()


# In[26]:


test_df['Vintage_year'].describe()

# test_df['Vintage_year'].value_counts()


# In[27]:


df['Vintage_year'].describe()


# In[28]:


df['Vintage_year'].value_counts()


# In[29]:


df['Product_Holding_B1'] = df['Product_Holding_B1'].apply(lambda x : literal_eval(x))
df['Product_Holding_B2'] = df['Product_Holding_B2'].apply(lambda x : literal_eval(x))

test_df['Product_Holding_B1'] = test_df['Product_Holding_B1'].apply(lambda x : literal_eval(x))

df.info()


# In[30]:


df.corr()


# In[31]:


Product_Holding_B1_list = df['Product_Holding_B1'].tolist()

Product_Holding_B1_flat_list = [item for sublist in Product_Holding_B1_list for item in sublist]
print(len(Product_Holding_B1_flat_list))
Product_Holding_B1_flat_list = list(set(Product_Holding_B1_flat_list))
print(len(Product_Holding_B1_flat_list))
print(Product_Holding_B1_flat_list)

b1_col_list = ['B1_'+item for item in Product_Holding_B1_flat_list]
for item in b1_col_list:
    df[item] = 0
    test_df[item] = 0


df.head(2)


# In[32]:


Product_Holding_B2_list = df['Product_Holding_B2'].tolist()

Product_Holding_B2_flat_list = [item for sublist in Product_Holding_B2_list for item in sublist]
print(len(Product_Holding_B2_flat_list))
Product_Holding_B2_flat_list = list(set(Product_Holding_B2_flat_list))
print(len(Product_Holding_B2_flat_list))
print(Product_Holding_B2_flat_list)

b2_col_list = ['B2_'+item for item in Product_Holding_B2_flat_list]
for item in b2_col_list:
    df[item]= 0

print(df.shape)
df.head(2)


# In[33]:


set(b1_col_list) not in set(b2_col_list)


# In[34]:



for i,row in df.iterrows():
  for item in row['Product_Holding_B1'] :
      df.loc[i, 'B1_'+item] = 1
  for item in row['Product_Holding_B2'] :
      df.loc[i, 'B2_'+item] = 1

df.head(2)


# In[36]:


df.columns


# In[37]:



for i,row in test_df.iterrows():
  for item in row['Product_Holding_B1'] :
      test_df.loc[i, 'B1_'+item] = 1

test_df.head(2)


# In[38]:


df['current_holdings'] = df['Product_Holding_B1'].apply(lambda x: len(x))
df = df.drop(columns = ['Customer_ID','Product_Holding_B1','Product_Holding_B2','Age','Vintage'])
df.head(2)


# In[39]:


test_df['current_holdings'] = test_df['Product_Holding_B1'].apply(lambda x: len(x))

test_df = test_df.drop(columns = ['Product_Holding_B1','Age','Vintage'])
test_df.head(2)


# In[40]:


test_df.columns


# In[41]:


df.columns


# In[42]:


df.shape


# In[43]:


test_df.shape


# In[44]:



df.to_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/preprocessed_train_data.csv', index=False)
test_df.to_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/preprocessed_test_data.csv', index=False)


# In[2]:


df = pd.read_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/preprocessed_train_data.csv')
print(df.shape)
# print(df.describe())
df.head(2)


# In[3]:


test_df = pd.read_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/preprocessed_test_data.csv')
print(test_df.shape)
test_df.head(2)


# In[4]:


Product_Holding_B2_flat_list = ['P6', 'P13', 'P14', 'P10', 'P9', 'P5', 'P4', 'P7', 'P20', 'P18', 'P00', 'P11', 'P16', 'P2', 'P15', 'P3', 'P12', 'P17', 'P1', 'P8']

b2_col_list = ['B2_'+item for item in Product_Holding_B2_flat_list]
len(b2_col_list)


# In[5]:



Product_Holding_B1_flat_list = ['P6', 'P21', 'P13', 'P14', 'P10', 'P9', 'P5', 'P4', 'P7', 'P20', 'P19', 'P18', 'P00', 'P11', 'P16', 'P2', 'P15', 'P3', 'P12', 'P17', 'P1', 'P8']

b1_col_list = ['B1_'+item for item in Product_Holding_B1_flat_list]
len(b1_col_list)


# In[6]:


from sklearn.model_selection import train_test_split

df = df.sample(frac=1).reset_index(drop=True)

X_train = df.drop(columns = b2_col_list)
y_train = df[b2_col_list]

print(X_train.shape)
print(y_train.shape)

train_one_fold_x, valid_one_fold_x , train_one_fold_y, valid_one_fold_y = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print('train_one_fold_x : ', train_one_fold_x.shape)
print('train_one_fold_y : ', train_one_fold_y.shape)
print('valid_one_fold_x : ', valid_one_fold_x.shape)
print('valid_one_fold_y : ', valid_one_fold_y.shape)

# for cols in y_train.columns:
#     print(cols)
#     print(y_train[cols].unique())


# In[7]:


test_df.head(2)


# In[8]:


X_train.columns


# In[9]:


train_one_fold_y.head(2)


# In[10]:


train_one_fold_x.columns


# In[14]:


valid_one_fold_x.head(2)


# In[78]:


#from sklearn.multioutput import MultiOutputClassifier
#from sklearn.ensemble import RandomForestClassifier

#forest = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, min_samples_split=5)
#multilabel_model = MultiOutputClassifier(forest)
#multilabel_model.fit(train_one_fold_x, train_one_fold_y)



import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
xgb_estimator = xgb.XGBClassifier(objective='multi:softprob',num_class=20,n_estimators =300,eval_metric='map',tree_method='gpu_hist',max_depth=9,learning_rate=0.02,subsample=0.7,verbose=1,scale_pos_weight=9)
multilabel_model = MultiOutputClassifier(xgb_estimator)
multilabel_model.fit(train_one_fold_x, train_one_fold_y)
#print(multilabel_model.score(valid_one_fold_x, valid_one_fold_y))




# In[ ]:


multilabel_model


# In[17]:


valid_pred = multilabel_model.predict(valid_one_fold_x)

valid_pred_df = pd.DataFrame(valid_pred,columns=valid_one_fold_y.columns)
print(valid_pred_df.shape)
valid_pred_df.head(2)


# In[19]:


valid_one_fold_y.head(2)


# In[18]:


valid_df = pd.concat([valid_one_fold_x, valid_one_fold_y], axis=1)
print(valid_df.shape)
valid_df.head()


# In[35]:


total_cols = valid_one_fold_y.columns
total_cols


# In[36]:


def get_labels(row):
    labels = []
    max_val = -1
    max_label = ''
    label_dict = {}
    for col in total_cols:
        if row[col] > max_val:
            max_val = row[col]
            max_label = col.split('_')[-1] 

        if row[col] > 0.5:
            labels.append(col.split('_')[-1])
            label_dict[col.split('_')[-1]] = row[col]
    
    if len(labels) == 0:
        labels.append(max_label)
        
    if len(labels) > 3:
        sorted_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        labels = []
        labels.extend(list(sorted_dict.keys())[0:3])
        
    return str(labels)


# In[37]:


valid_pred_df['Product_Holding_B2'] = valid_pred_df.apply(get_labels, axis=1 )
print(valid_pred_df.shape)
valid_pred_df.head()


# In[38]:


valid_one_fold_y['Product_Holding_B2'] = valid_one_fold_y.apply(get_labels, axis=1 )
print(valid_one_fold_y.shape)
valid_one_fold_y.head()


# In[39]:


valid_one_fold_y['Product_Holding_B2'] = valid_one_fold_y['Product_Holding_B2'].apply(lambda x : literal_eval(x))
valid_pred_df['Product_Holding_B2'] = valid_pred_df['Product_Holding_B2'].apply(lambda x : literal_eval(x))


# In[41]:


valid_one_fold_y.reset_index(inplace=True, drop=True)
valid_pred_df.reset_index(inplace=True, drop=True)


# In[43]:


valid_one_fold_y['Product_Holding_B2'][0]
valid_pred_df['Product_Holding_B2'][0]


# In[66]:


mean_avg_prec = 0

for i, row in valid_pred_df.iterrows() :
    index = 0
    total_index = 0 
    prec = 0
#     print(row['Product_Holding_B2'])
    for item in row['Product_Holding_B2'] :
#         print(index, item)
        total_index += 1
#         print(valid_one_fold_y.loc[i,'Product_Holding_B2'])
#         print(valid_one_fold_y.loc[i,'Product_Holding_B2'][index])
        if len(valid_one_fold_y.loc[i,'Product_Holding_B2']) > index :
            if valid_one_fold_y.loc[i,'Product_Holding_B2'][index] == item :
                if index == 0 :
                    prec = 1
                else :
                    prec += 1*(index/total_index)
        else :
            break
        index+=1
#     print("precision = ", prec)
    avg_prec = prec/3
#     print("avg precision = ", prec/3)
    mean_avg_prec += avg_prec
#     break

print("mean_avg_prec : ", mean_avg_prec/len(valid_df)*100)


# In[63]:


avg_prec


# In[64]:


len(valid_df)


# In[28]:


# total_b1_cols = b1_col_list
# total_b1_cols


# In[29]:


# def get_b1_labels(row):
#     labels = []
#     max_val = -1
#     max_label = ''
#     label_dict = {}
#     for col in total_b1_cols:
#         if row[col] > max_val:
#             max_val = row[col]
#             max_label = col.split('_')[-1] 

#         if row[col] > 0.5:
#             labels.append(col.split('_')[-1])
#             label_dict[col.split('_')[-1]] = row[col]
    
#     if len(labels) == 0:
#         labels.append(max_label)
        
#     if len(labels) > 3:
#         sorted_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))
#         print(sorted_dict)
#         labels = []
#         labels.extend(list(sorted_dict.keys())[0:3])
        
#     return str(labels)


# In[30]:


# valid_df['Product_Holding_B1'] = valid_df.apply(get_b1_labels, axis=1 )
# print(valid_df.shape)
# valid_df.head()


# In[31]:


# valid_df['Product_Holding_B1'] = valid_df['Product_Holding_B1'].apply(lambda x : literal_eval(x))


# In[32]:


# valid_pred_df['Product_Holding_B2'] = valid_pred_df['Product_Holding_B2'].apply(lambda x : literal_eval(x))


# In[33]:


# valid_df[['Product_Holding_B2','Product_Holding_B1']]


# In[34]:


# valid_pred_df['Product_Holding_B2']


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


# valid_one_fold_y


# In[66]:


# from sklearn.metrics import average_precision_score
# from sklearn.metrics import multilabel_confusion_matrix

# from sklearn.metrics import confusion_matrix

# print(confusion_matrix(valid_one_fold_x, valid_one_fold_y))


# In[ ]:





# In[69]:


# !pip install xgboost


# In[70]:


# import xgboost as xgb

# # create XGBoost instance with default hyper-parameters
# xgb_estimator = xgb.XGBClassifier(objective='rank:map')

# multilabel_model = MultiOutputClassifier(xgb_estimator)
# multilabel_model.fit(train_one_fold_x, train_one_fold_y)

# print(multilabel_model.score(valid_one_fold_x, valid_one_fold_y))


# In[64]:


# train_one_fold_y


# In[71]:


# from sklearn.linear_model import SGDClassifier

# sgd_linear_clf = SGDClassifier(random_state=1, max_iter=5)
# mor = MultiOutputClassifier(sgd_linear_clf, n_jobs=4)

# mor.fit(train_one_fold_x, train_one_fold_y)

# print(mor.score(valid_one_fold_x, valid_one_fold_y))


# In[67]:


test_df.columns


# In[68]:


test_df = test_df.set_index('Customer_ID')
test_df.head()


# In[69]:


preds = multilabel_model.predict(test_df)
preds


# In[70]:


df_result= pd.DataFrame(preds,columns=b2_col_list)
print(df_result.shape)
df_result.head(2)


# In[71]:


test_df.shape


# In[72]:


test_df['Customer_ID'] = test_df.index
test_df.reset_index(inplace=True, drop=True)
print(test_df.shape)
test_df.head()


# In[73]:


test_df = pd.concat([test_df, df_result], axis=1)
print(test_df.shape)
test_df.head()


# In[74]:


total_cols = df_result.columns
total_cols


# In[75]:


def get_labels(row):
    labels = []
    max_val = -1
    max_label = ''
    label_dict = {}
    for col in total_cols:
        if row[col] > max_val:
            max_val = row[col]
            max_label = col.split('_')[-1] 

        if row[col] > 0.5:
            labels.append(col.split('_')[-1])
            label_dict[col.split('_')[-1]] = row[col]
    
    if len(labels) == 0:
        labels.append(max_label)
        
    if len(labels) > 3:
        sorted_dict = dict(sorted(label_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        labels = []
        labels.extend(list(sorted_dict.keys())[0:3])
        
    return str(labels)


# In[76]:


test_df['Product_Holding_B2'] = test_df.apply(get_labels, axis=1 )
print(test_df.shape)
test_df.head()


# In[77]:


test_df[['Customer_ID', 'Product_Holding_B2']].to_csv('E:/WorkSpace -Personal/HACKATHONS/AmExpert2021/Data/submission_randomforest_entropy.csv', index=False)


# In[ ]:


#auc_y1 = roc_auc_score(ytest[:,0],yhat[:,0])
#auc_y2 = roc_auc_score(ytest[:,1],yhat[:,1])
 
#print("ROC AUC y1: %.4f, y2: %.4f" % (auc_y1, auc_y2))
#ROC AUC y1: 0.9206, y2: 0.9202


# In[ ]:





# In[50]:


# from sklearn.metrics import accuracy_score
# print('Accuracy on test data: {:.1f}%'.format(accuracy_score(valid_one_fold_y, multilabel_model.predict(valid_one_fold_x))*100))
# df_test = df_test.drop(columns= B2_cols)
# preds = multilabel_model.predict(df_test)


# In[24]:


# from sklearn.multioutput import MultiOutputClassifier
# import catboost

# classifier = MultiOutputClassifier(catboost.CatBoostClassifier(verbose=1, iterations=20))
# # classifier.fit(X_train, y_train, cat_features=list(X_train.columns))

# classifier.fit(X_train, y_train, cat_features=list(X_train.columns))


# In[51]:


# test_df.sort_values(by='Customer_ID')

