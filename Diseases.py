#!/usr/bin/env python
# coding: utf-8

# CDC NNDSS - Table I. infrequently reported notifiable diseases 
# Data for this can be viewed and downloaded at https://healthdata.gov/dataset/nndss-table-i-infrequently-reported-notifiable-diseases-1
# 
# The purpose of this data is to analyze the top 3-5 diseases throughout all twelve months and years 2013-2017. 

# In[585]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import datetime

get_ipython().run_line_magic('matplotlib', 'inline')


# In[586]:


# Read the csv file
df18 = pd.read_csv("18_NNDSS.csv")
df18.head()


# In[587]:


# Clean up the column headers
df18.columns = df18.columns.str.strip().str.lower().str.replace('  ', ' ').str.replace(' ', '_').str.replace('(', '').str.replace(')', '')


# In[588]:


# Delete unnecessary columns
columns_to_drop = ['current_week', 'current_week,_flag', 'cum_2018', 'cum_2018,_flag', '5-year_weekly_average§', 
                   '5-year_weekly_average§,_flag', 'total_cases_reported_for_pervious_years_2017,_flag', 
                   'total_cases_reported_for_pervious_years_2016,_flag', 
                   'total_cases_reported_for_pervious_years_2015,_flag', 
                   'total_cases_reported_for_pervious_years_2014,_flag', 
                   'total_cases_reported_for_pervious_years_2013,_flag', 
                   'states_reporting_cases_during_current_week_no.']

df18.drop(columns_to_drop, inplace=True, axis=1)


# In[589]:


# Delete non ASCII characters
df18['disease'] = df18['disease'].apply(lambda x: ''.join([" " if (ord(i) < 32 or ord(i) > 126) or ord(i) == 42  else i for i in x]))


# In[590]:


df18.head()


# In[591]:


# Shorten some of the column names
df18.columns = df18.columns.str.replace('_reported_for_pervious_years_', '_')


# In[592]:


#Reorder columns
df18 = df18[[ 'disease','mmwr_week', 'total_cases_2013', 'total_cases_2014', 'total_cases_2015', 'total_cases_2016', 'total_cases_2017']]
df18


# In[593]:


df18.info()


# In[594]:


# Replace null values with 0
cols = ['mmwr_week', 'total_cases_2017', 'total_cases_2016', 'total_cases_2015', 
                'total_cases_2014', 'total_cases_2013']


df18[cols] = df18[cols].fillna(0)
#astype(int)


# In[595]:


df18.isnull().sum()


# In[596]:


df18.info()


# In[597]:


# Check for duplicate rows
sum(df18.duplicated())


# In[598]:


has_duplicate = df18.duplicated()

duplicates = df18[has_duplicate]

duplicates


# In[599]:


#Week 32 has duplicate data, need to drop duplicates
df18.drop_duplicates(inplace = True)

sum(df18.duplicated())


# In[600]:


# Filter out diseases with 0 cases between 2013-2017
df18 = df18[(df18.total_cases_2017 > 0) | (df18.total_cases_2016 > 0) | (df18.total_cases_2015 > 0)| (df18.total_cases_2014 > 0) & (df18.total_cases_2013 > 0)]


# In[601]:


df18.shape


# In[602]:


#Stripping white spaces
df18.disease = df18.disease.str.strip()


# In[603]:


# Added all cases for all weeks per disease in a new dataframe
df18_sum = df18.groupby('disease').sum()
df18_sum


# In[604]:


# Add a column for total cases
df18_sum['case_total'] = df18_sum['total_cases_2017'] + df18_sum['total_cases_2016'] + df18_sum['total_cases_2015'] + df18_sum['total_cases_2014'] + df18_sum['total_cases_2013']
df18_sum


# In[605]:


#Sorted by largest case total value
top_df18_yrs = df18_sum.sort_values('case_total', ascending=False).reset_index().head(5)
top_df18_yrs


# In[606]:


#Created variables for each top disease
Listeriosis = top_df18_yrs.iloc[0,7]
Cyclosporiasis = top_df18_yrs.iloc[1,7]
Congenital_syphilis = top_df18_yrs.iloc[2,7]
Typhoid_fever = top_df18_yrs.iloc[3,7]
Hemolytic_uremic = top_df18_yrs.iloc[4,7]


# In[607]:


#Bar chart for each top disease
plt.figure(figsize = [13, 9])
locations = [1, 2, 3, 4, 5]
heights = [Listeriosis, Cyclosporiasis, Congenital_syphilis, Typhoid_fever, Hemolytic_uremic]
labels = ['Listeriosis', 'Cyclosporiasis', 'Syphilis (Congenital)', 'Typhoid Fever', 'Hemolyitic Uremic Syndrome']
plt.bar(locations, heights, tick_label=labels, align = 'center')
plt.xticks(rotation=0)
plt.title('Top CDC Notifiable Diseases Between 2013 and 2017', fontsize = 20)
plt.xlabel('Disease', fontsize = 16)
plt.ylabel('Count', fontsize = 16);


# The top 5 diseases according to the CDC are: Liseriosis, Cyclosporiasis, Syphilis, Typhoid Fever, and Hemolytic Uremia.

# In[608]:


#Chart of diseases by year
top_df18_yrs = pd.DataFrame({
    '2013': [38220.0, 39984.0, 17748.0, 17238.0, 16779.0],
    '2014': [39988.0, 19788.0, 23358.0, 17799.0, 12750.0],
    '2015': [39936.0, 32895.0, 25143.0, 18717.0, 13974.0],
    '2016': [40872.0, 27285.0, 32283.0, 19176.0, 15657.0],
    '2017': [44732.0, 59005.0, 43156.0, 19737.0, 16100.0]},
    index= ['Listeriosis', 'Cyclosporiasis', 'Syphilis (Congenital)', 'Typhoid Fever', 'Hemolytic Uremia'])
top_df18_yrs.T.plot(figsize = [12, 8]);


# Listeriosis seems to maintain a high level of occurence throughout the years, without much fluctuation. Cyclosporiasis has a more fluctuation, starting around 40,000 in 2013, then decreasing to 20,000 the next year, and continues to have peaks and valleys until spiking up to approximately 60,000 in 2017.

# In[609]:


# Original clean dataframe
df18


# In[610]:


#Copy original clean dataframe
df18_months = df18
df18_months


# In[611]:


#Added a year column to calculate month
df18_months['year_2013'] = 2013
df18_months


# In[612]:


#Calculated month and year in new column
df18_months['week_year'] = df18.year_2013*1000+df18.mmwr_week*7-6 
df18['month'] = pd.to_datetime(df18['week_year'], format='%Y%j')

df18_months


# In[613]:


#Added a month column based on new date
df18_months['month'] = pd.DatetimeIndex(df18_months['month']).month

df18_months


# In[614]:


#Reorder columns in new dataset
df18_months = df18_months[['disease', 'month', 'total_cases_2013', 'total_cases_2014', 'total_cases_2015', 'total_cases_2016', 'total_cases_2017']]
df18_months


# In[ ]:





# In[615]:


#Filtered the top 5 disease from previous analyses
df18_months = df18_months[df18_months["disease"].isin(["Listeriosis", "Cyclosporiasis", "Syphilis, congenital", "Typhoid fever", "Hemolytic uremic syndrome, postdiarrheal"])]
df18_months


# In[616]:


#Filtered out disease totals by month
df18_months = df18_months.groupby(['month', 'disease'], as_index=False).sum()
df18_months


# In[618]:


#Plot top diseases by month
fig, ax = plt.subplots(figsize=(8,5))
ax=sns.lineplot(data=df18_months, x='month', y='total_cases_2013', hue='disease')
ax.set(ylim=(900, 6000))

fig, ax = plt.subplots(figsize=(8,5))
ax=sns.lineplot(data=df18_months, x='month', y='total_cases_2014', hue='disease')
ax.set(ylim=(900, 6000))

fig, ax = plt.subplots(figsize=(8,5))
ax=sns.lineplot(data=df18_months, x='month', y='total_cases_2015', hue='disease')
ax.set(ylim=(900, 6000))

fig, ax = plt.subplots(figsize=(8,5))
ax=sns.lineplot(data=df18_months, x='month', y='total_cases_2016', hue='disease')
ax.set(ylim=(900, 6000))

fig, ax = plt.subplots(figsize=(8,5))
ax=sns.lineplot(data=df18_months, x='month', y='total_cases_2017', hue='disease')
ax.set(ylim=(900, 6000));


# In[ ]:




