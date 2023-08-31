#################################
##### Testing the functions #####
#################################

### importing packages
import pandas as pd
import numpy as np
import random
# import statsmodels.formula.api as smf
from collections import Counter
from pandas.api.types import is_numeric_dtype
# import datetime
# from datetime import date
from numpy.random import choice
# from random import choices

# Define file paths
user_file_path = 'H:\\My Drive\\sync\\Green Pheasants\\Data and code\\Real data\\users.csv'
item_file_path = 'H:\\My Drive\\sync\\Green Pheasants\\Data and code\\Real data\\items.csv'
interaction_file_path = 'H:\\My Drive\\sync\\Green Pheasants\\Data and code\\Real data\\interactions.csv'

def load_data(path):
    df = pd.read_csv(path)
    print(df.head())
    print(df.info())
    return df

df_users = load_data(user_file_path)
df_items = load_data(item_file_path)
df_interactions = load_data(interaction_file_path)

########################################
### a random recommendation function ###
########################################
 
itemids = df_items['itemid'].unique()
random_itemid=function_random(itemids)
print(random_itemid)

##########################################################
### remove rows with missing values in df_interactions ###
##########################################################

# Remove rows with missing values in columns needed for the recommendation code
df_interactions = function_remove_rows_with_missing_values(df_interactions)

# examine df_interactions
print(df_interactions.shape)
print(df_interactions.info())
print(df_interactions.head())
print(df_interactions['creatorid'].head())

###################################################
### create necessary cloumns in df_interactions ###
###################################################

df_interactions=function_add_columns(df_interactions)
# print(df_interactions.info())
print(df_interactions.head())

#######################################################################################
###### Create and train the 'unique_residuals' recommendation function        #########
#######################################################################################

df_interactions_test = function_unique_residuals_model(df_interactions)

# explore the output
# select the last 5 columns of df_interactions (these are the new columns that the function created)
df_interactions_for_test = df_interactions.iloc[:, -5:]

# show the first 10 rows of the new columns
print(df_interactions_for_test.head(10))

##################################################################
###### A function that filters a dataframe by theme or mood ######
##################################################################

print(df_interactions['imood1'].head)
print(df_interactions['imood2'].head)
print(df_interactions['imood3'].head)
print(df_interactions['itheme1'].head)
print(df_interactions['itheme2'].head)
print(df_interactions['itheme3'].head)
df_interactions_filtered = function_df_filter_theme_mood(df_interactions, theme='Love', mood='all')
print(df_interactions_filtered.shape)
print(df_interactions.shape)

############################################################################################################################
###### A function that attaches userid and characteristics to each item, creating all combinations of users and items ######
############################################################################################################################

print(df_users.head())
print(df_items.head())

df_users_items = function_attach_userids_to_items(df_users, df_items)
print(df_users_items.info())
print(df_users_items.shape)
print(df_users_items.head())
print(df_users_items.iloc[100:110]) # examine some rows in the middle of the dataframe
print(df_users.shape[0]) # print the number of unique users
print(df_items.shape[0]) # print the number of unique items
print(df_users.shape[0]*df_items.shape[0]) # print the number of unique user-item combinations
print(df_users_items.shape[0]) # print the number of rows in df_users_items

# check whether the number of rows in df_user_items 
# is the product of the number of unique users and the number of unique items
if df_users_items.shape[0]==df_users.shape[0]*df_items.shape[0]:
    print("It seems to have worked well. The number of rows in df_users_items is the product of the number of rows df_users and df_items")
else: print("Something went wrong")

# Count missing values in each column
missing_values = df_users_items.isna().sum()
print(missing_values)

##################################################################################
###### A function that removes items that the users have viewed in the past ######
##################################################################################

# create df_users_items
df_users_items = function_attach_userids_to_items(df_users, df_items)

### concatenate the userid and itemid columns, in df_users_items and in df_interactions
# in df_interactions
df_interactions['userid_itemid']=(df_interactions['userid'].astype(str) + "_" + df_users_items['itemid'].astype(str))

# in df_users_items
df_users_items['userid_itemid']=(df_users_items['userid'].astype(str) + "_" + df_users_items['itemid'].astype(str))
print(df_users_items.shape)

len(df_users_items['userid'].astype(str))
len(df_users_items['itemid'].astype(str))

# Count missing values in each column
missing_userids=df_users_items['userid'].isna().sum()
print(missing_values)

# Count the number of userid_itemid that appear in both the interactions df and the users_items df
# Convert the columns to set, perform intersection, count the number of elements
common_values = len(set(df_interactions['userid_itemid']).intersection(set(df_users_items['userid_itemid'])))

df_interactions['userid_itemid']
df_users_items['userid_itemid']

print(f"The number of identical values is: {common_values}")

# Apply the function
df_users_items_test=function_removing_viewed_items(df_interactions, df_users_items)
print(df_users_items_test.shape)
print(df_users_items.shape)

#################################################################################################################
###### A function that removes poems written by a poet that wrote at least one poem                       #######
###### that the user has added to their collection in the past,                                            #######
###### for  those who selected that they would would like to expand their taste, or alternate between     #######
###### fitting and expanding                                                                              #######
#################################################################################################################

# creating a test df
df_users_items_test=function_removing_items_to_expand_taste(df_interactions, df_users_items)

# checking the shape of the test df before and after filtering
print(df_users_items.shape)
df_users_items_test=function_removing_items_to_expand_taste(df_interactions, df_users_items)
print(df_users_items_test.shape)

###################################################################################
###### A function that calculates probabilities of showing items to visitors ######
###################################################################################

# applying the function
df_items_with_betas_test = function_calculate_probabilities_visitors(df_interactions, df_items, theme='all', mood='Gloomy')

# removing "test" from its name
df_items_with_betas = df_items_with_betas_test

# examine the resulting df
print(df_items_with_betas.head())
print(df_items_with_betas.info())

##########################################################
###### A function that adds betas to df_users_items ######
##########################################################

# apply the function
df_users_items_with_betas_test = function_calculate_probabilities_users(df_users, df_items, df_interactions)

# examine the resulting df
print(df_users_items_with_betas_test.head())
print(df_users_items_with_betas_test.info())

########################################################################################
#### A function that calculates recommendation probabilities for one online visitor ####
########################################################################################

# create df_items_with_betas_test
df_items_with_betas_test = function_calculate_recommendation_probabilities_one_visitor(df_items_with_betas, theme='all', mood='all')

# creating an identical dataframe without the word 'test'
df_items_with_betas = df_items_with_betas_test

# apply the function
df_with_final_predictions_test = function_calculate_recommendation_probabilities_one_visitor(df_items_with_betas_test, theme='all', mood='all')

# examine the resulting df
print(df_with_final_predictions_test.head())
print(df_with_final_predictions_test.info())

##############################################################################
#### A function that calculates recommendation probabilities for one user ####
##############################################################################

# create df_items_with_betas_test 
df_users_items_with_betas_test = function_calculate_probabilities_users(df_users, df_items, df_interactions)

### create df_users_requesting_recommendation

# Get unique values
unique_values = df_users['userid'].unique()

# Create a subset. In this case, we'll take the first 1 unique values.
# Adjust this to get the subset size you want.
subset = unique_values[:5]

# Create new series from subset of unique values
df_users_requesting_recommendation_test = pd.Series(subset)

# convert the series to a dataframe
df_users_requesting_recommendation_test = df_users_requesting_recommendation_test.to_frame()

# change column name to 'userid'
df_users_requesting_recommendation_test.columns = ['userid']

# create an identical dataframe without the word 'test'
df_users_requesting_recommendation=df_users_requesting_recommendation_test

# apply the function
df_with_final_predictions_test = function_calculate_recommendation_probabilities_one_user(df_users_requesting_recommendation_test, df_users_items_with_betas_test, theme='all', mood='all')

# examine the resulting df
print(df_with_final_predictions_test.head())
print(df_with_final_predictions_test.info())

###########################################################################################
###### A function that chooses an item to show a visitor or a user, after            ######
###### the previous function has added the show probabilities.                       ######
###### It chooses one out of two recommendation methods, randomly:                   ######
###### completely random recommendation or based on the 'unique residuals' model     ######
###########################################################################################

# apply the function
df_results_test = function_choose_one_item_to_display(df_with_final_predictions_test)

# examine the resulting df
print(df_results_test.head())

########################################################################################
#### A function that calculates recommendation probabilities for many offline users ####
########################################################################################

# set userid
userids=df_users['userid'].unique()
random_userid=function_random(userids)    
print(random_userid)

# apply the function
df_recommendations_test = function_choose_items_to_display_for_multiple_users(df_users_items_with_betas, df_users_requesting_recommendation)

# examine the resulting df
df_recommendations_test.head()

