###############################################################
##### Functions for Green Pheasants recommendation system #####
###############################################################

### Import the data
"""
Import three datasets from the database: "users", "items" and "interactions".
Turn them into Pandas dataframes and name them: df_users, df_items and df_interactions
In the Github repository there are three sample csv files that can be downloaded to your 
local machine and imported into your IDE in order to test the code.
"""

########################################
### a random recommendation function ###
########################################

def function_random(itemids): 
    itemid = random.choice(itemids)
    return(itemid)

"""
Defintions that are used throughout this script
User - someone who has a user account. 
Visitor - someone who does not have a user account, or has one but is not logged in.
"""

##########################################################
### remove rows with missing values in df_interactions ###
##########################################################

def function_remove_rows_with_missing_values(df_interactions):

    # Remove rows with missing values in columns needed for the recommendation code
    # List of columns to check for missing values
    cols_to_check = ['userid', 'itemid', 'creatorid', 'inum_words_bin']

    # Remove the rows with missing values in specified columns
    df_interactions = df_interactions.dropna(subset=cols_to_check)
    
    return(df_interactions)

###################################################
### create necessary cloumns in df_interactions ###
###################################################

"""
This function adds the interaction between the userid and the creatorid,
to account for users' preference towards certain poets
"""

def function_add_columns(df_interactions):
    
    # tell pandas that I indeed want to modify the interactions DataFrame
    df_interactions = df_interactions.copy()

    # concatenating the userid and creatorid columns into one numerical column called "userid_creatorid"
    df_interactions['userid_creatorid'] = df_interactions['userid'].astype(str) + "_" + df_interactions['creatorid'].astype(str).apply(lambda x: str(int(x)))

    ### create the userid_num_words_bin variable by concatenating the two columns
    # concatenate the strings
    df_interactions['userid_inum_words_bin'] = df_interactions['userid'].astype(str) + "_" + df_interactions['inum_words_bin']

    return(df_interactions)

#######################################################################################
###### Create and train the 'unique_residuals' recommendation function        #########
#######################################################################################

def function_unique_residuals_model(df_interactions): 
    
    # calculate the mean of the collection column in the interactions df
    mu=np.nanmean(df_interactions['collection'])
    
    # add the necessary columns
    df_interactions=function_add_columns(df_interactions)
    
    ## adding user-specific intercepts
    # calculating the intercepts
    means_userid=df_interactions.groupby(['userid']).mean()
    means_userid['b_userid']=means_userid['collection']-mu
    
    # adding the intercepts to the 'df_interactions' dataframe
    df_interactions = pd.merge(df_interactions,means_userid['b_userid'],on='userid', how='left')

    ## adding item-specific intercepts
    # calculating the intercepts
    means_itemid=df_interactions.groupby(['itemid']).mean()
    means_itemid['b_itemid']=(means_itemid['collection']
                              -means_itemid['b_userid']
                              -mu)

    # adding the intercepts to the 'df_interactions' dataframe
    df_interactions = pd.merge(df_interactions,means_itemid['b_itemid'],on='itemid', how='left')


    ## adding creator-specific intercept ('creator' means the poet)
    # calculating the intercepts
    means_creatorid=df_interactions.groupby(['creatorid']).mean()
    means_creatorid['b_creatorid']=(means_creatorid['collection']
                                    -means_creatorid['b_userid']
                                    -means_creatorid['b_itemid']
                                    -mu)

    # adding the intercepts to the 'df_interactions' dataframe
    df_interactions = pd.merge(df_interactions,means_creatorid['b_creatorid'],on='creatorid', how='left')
    
    ## adding the interaction between the userid and the creator,
    ## to account for users' preference towards certain poets

    # calculating the intercepts
    means_userid_creatorid=df_interactions.groupby(['userid_creatorid']).mean()
    means_userid_creatorid['b_userid_creatorid']=(means_userid_creatorid['collection']
                                                  -means_userid_creatorid['b_userid']
                                                  -means_userid_creatorid['b_itemid']
                                                  -means_userid_creatorid['b_creatorid']
                                                  -mu)

    # adding the intercepts to the 'df_interactions' dataframe
    df_interactions = pd.merge(df_interactions,means_userid_creatorid['b_userid_creatorid'],on='userid_creatorid', how='left')
    return(df_interactions)

##################################################################
###### A function that filters a dataframe by theme or mood ######
##################################################################
"""
The user chooses theme or mood, they can't choose both
This function then filters the interactions dataframe, so that it would include
only items with the them or with the mood that the user chose.
"""
def function_df_filter_theme_mood(df_interactions, theme='all', mood='all'):
    
    # Create the dataframe that will be filtered
    df_filtered_theme_mood = df_interactions

    if theme != 'all':
            theme_cols = ['itheme1', 'itheme2', 'itheme3', 'itheme4', 'itheme5']
            df_filtered_theme_mood = df_filtered_theme_mood[df_filtered_theme_mood[theme_cols].isin([theme]).any(axis=1)]
            
    if mood != 'all':
            mood_cols = ['imood1', 'imood2', 'imood3']
            df_filtered_theme_mood = df_filtered_theme_mood[df_filtered_theme_mood[mood_cols].isin([mood]).any(axis=1)]
    
    return(df_filtered_theme_mood)

############################################################################################################################
###### A function that attaches userid and characteristics to each item, creating all combinations of users and items ######
############################################################################################################################

def function_attach_userids_to_items(df_users, df_items):

    ### Concatenating users_requesting_recommendation and df_items
   # Validation
   assert 'userid' in df_users.columns, "df_users must contain 'userid' column"
   assert 'itemid' in df_items.columns, "df_items must contain 'itemid' column"
   assert 'creatorid' in df_items.columns, "df_items must contain 'creatorid' column"
    
   # Create the Cartesian product of users and items
   df_users_items = df_users[['userid']].merge(df_items[['itemid', 'creatorid']], how='cross')

   return(df_users_items)

##################################################################################
###### A function that removes items that the users have viewed in the past ######
##################################################################################
"""
These items are removed so users will not get a recommendation
for a poem that they already viewed.
"""

def function_removing_viewed_items(df_interactions, df_users_items):

    ### concatenating the userid and itemid columns, in df_users_items and in df_interactions
    # in df_interactions
    df_interactions['userid_itemid']=(df_interactions['userid'].astype(str) + "_" + df_interactions['itemid'].astype(str))
    # in df_users_items
    df_users_items['userid_itemid']=(df_users_items['userid'].astype(str) + "_" + df_users_items['itemid'].astype(str))

    ### removing the userid_itemid combinations in df_users_items that appear in df_interactions
    ### so that users will not be presented with items that they have already viewed
    df_users_items = df_users_items[~df_users_items['userid_itemid'].isin(df_interactions['userid_itemid'])]

    return(df_users_items)

#################################################################################################################
###### A function that removes poems written by a poet that wrote at least one poem                       #######
###### that the user has added to their collection in the past,                                            #######
###### for  those who selected that they would would like to expand their taste, or alternate between     #######
###### fitting and expanding                                                                              #######
#################################################################################################################
"""
# Technically, this function removes items from df_users_items that have the same userid_creatorid combination 
# as in df_interactions (after filtering)
"""
def function_removing_items_to_expand_taste(df_interactions, df_users_items):
    
    # Create a copy of the subset of df_interactions where 'collection' equals 1
    # The .copy() is used to ensure we do not modify the original df_interactions DataFrame
    df_interactions_added_to_collection = df_interactions.loc[df_interactions['collection']==1].copy()

    # Create a new column 'userid_creatorid' in df_interactions_added_to_collection
    # This is done by converting 'userid' and 'creatorid' to strings, concatenating them and converting the result to an integer
    df_interactions_added_to_collection['userid_creatorid']=(df_interactions_added_to_collection['userid'].astype(int).astype(str) + df_interactions_added_to_collection['creatorid'].astype(int).astype(str)).astype(int)

    # Create a copy of df_users_items and add a new column 'userid_creatorid'
    # This is done in the same way as for df_interactions_added_to_collection
    df_users_items=df_users_items.copy()
    df_users_items['userid_creatorid']=(df_users_items['userid'].astype(int).astype(str) + df_users_items['creatorid'].astype(int).astype(str)).astype(int)

    # Create a new dataframe df_users_items_filtered, which is a copy of df_users_items with the rows containing the common 'userid_creatorid' values removed
    df_users_items_filtered = df_users_items[~df_users_items['userid_creatorid'].isin(df_interactions_added_to_collection['userid_creatorid'])]

    # Return the filtered DataFrame
    return(df_users_items_filtered)

###################################################################################
###### A function that calculates probabilities of showing items to visitors ######
###################################################################################

def function_calculate_probabilities_visitors(df_interactions, df_items):
    
    # counting the number of times that each item participated in an interaction
    df_interactions['number_of_interactions'] = df_interactions.groupby('itemid')['itemid'].transform('count')
    
    # calculating the unique residuals 
    df_interactions_unique_residuals=function_unique_residuals_model(df_interactions)  

    # calculating the mean
    mu=np.nanmean(df_interactions_unique_residuals['collection'])

    # calculate the prediction of adding each item to the collection
    # without any information about the user
    df_interactions_grouped_by_itemid_means = df_interactions_unique_residuals.groupby(['itemid']).mean() # grouping by itemid and calculating the mean
    df_interactions_grouped_by_itemid_means = df_interactions_grouped_by_itemid_means.reset_index(level=0) # resetting the index

    df_interactions_grouped_by_creatorid_means = df_interactions_unique_residuals.groupby(['creatorid']).mean() # grouping by creatorid and calculating the mean
    df_interactions_grouped_by_creatorid_means = df_interactions_grouped_by_creatorid_means.reset_index(level=0) # resetting the index

    # adding the betas to df_items
    df_items_with_betas = pd.merge(df_items,df_interactions_grouped_by_itemid_means[['itemid', 'b_itemid', 'number_of_interactions']], on='itemid', how='left') # adding itemid betas
    df_items_with_betas = pd.merge(df_items_with_betas,df_interactions_grouped_by_creatorid_means[['creatorid', 'b_creatorid']],on='creatorid', how='left') # adding creatorid betas

    # replacing missing values of betas with zero
    df_items_with_betas[['b_itemid', 'b_creatorid']] = df_items_with_betas[['b_itemid', 'b_creatorid']].replace(np.nan, 0)

    # replacing missing values of number of interactions with zero
    df_items_with_betas['number_of_interactions'] = df_items_with_betas['number_of_interactions'].replace(np.nan, 0)

    # calculating the prediction
    df_items_with_betas['prediction_by_residuals']=(
        mu+
        df_items_with_betas['b_itemid']+
        df_items_with_betas['b_creatorid'])

    # replacing missing values with the mean
    df_items_with_betas['prediction_by_residuals'] = df_items_with_betas['prediction_by_residuals'].replace(np.nan, mu)

    ### assigning the probabilities of showing each item to the visitor 
    # determing the weight of the mean (if only one person has viewed the item, the prediction of showing it will be 
    # 50% the mean and 50% the prediction be residual, because information from one person is not very reilable)
    df_items_with_betas['weight_of_mean']=1/(df_items_with_betas['number_of_interactions']+1) 
    df_items_with_betas['prediction_weighted']=(mu*df_items_with_betas['weight_of_mean']+
                                    df_items_with_betas['prediction_by_residuals']*(1-df_items_with_betas['weight_of_mean']))

    # creating the prediction of showing each item
    # so that the sum of all probabilities is 1
    #sum_of_weighted_probabilities=df_items_with_betas['prediction_weighted'].sum()
    #df_items_with_betas['final_prediction']=df_items_with_betas['prediction_weighted']/sum_of_weighted_probabilities
    
    return(df_items_with_betas)

##########################################################
###### A function that adds betas to df_users_items ######
##########################################################

def function_calculate_probabilities_users(df_users, df_items, df_interactions):
    
    # counting the number of times that each item participated in an interaction
    df_interactions['number_of_interactions'] = df_interactions.groupby('itemid')['itemid'].transform('count')
    
    # calculating the unique residuals 
    df_interactions_unique_residuals=function_unique_residuals_model(df_interactions)  

    # calculating the mean
    mu=np.nanmean(df_interactions_unique_residuals['collection'])

    # calculating the betas
    df_interactions_grouped_by_userid_means = df_interactions_unique_residuals.groupby(['userid']).mean() # grouping by userid and calculating the mean
    df_interactions_grouped_by_userid_means = df_interactions_grouped_by_userid_means.reset_index(level=0) # resetting the index
    
    df_interactions_grouped_by_itemid_means = df_interactions_unique_residuals.groupby(['itemid']).mean() # grouping by itemid and calculating the mean
    df_interactions_grouped_by_itemid_means = df_interactions_grouped_by_itemid_means.reset_index(level=0) # resetting the index

    df_interactions_grouped_by_creatorid_means = df_interactions_unique_residuals.groupby(['creatorid']).mean() # grouping by creatorid and calculating the mean
    df_interactions_grouped_by_creatorid_means = df_interactions_grouped_by_creatorid_means.reset_index(level=0) # resetting the index

    df_interactions_grouped_by_userid_creatorid_means = df_interactions_unique_residuals.groupby(['userid_creatorid']).mean() # grouping by userid_creatorid and calculating the mean
    df_interactions_grouped_by_userid_creatorid_means = df_interactions_grouped_by_userid_creatorid_means.reset_index(level=0) # resetting the index
    
    # adding the userid to df_items
    df_users_items=function_attach_userids_to_items(df_users, df_items)

    # concatenating userid and creatorid
    df_users_items['userid_creatorid']=(df_users_items['userid'].astype(str)+df_users_items['creatorid'].astype(str)).astype(int)

    # removing items that users viewed in the past
    df_users_items_filtered=function_removing_viewed_items(df_interactions, df_users_items)

    # adding the betas to df_users_items
    df_users_items_with_betas = pd.merge(df_users_items_filtered,df_interactions_grouped_by_userid_means[['userid', 'b_userid']], on='userid', how='left') # adding itemid betas
    df_users_items_with_betas = pd.merge(df_users_items_with_betas,df_interactions_grouped_by_itemid_means[['itemid', 'b_itemid', 'number_of_interactions']], on='itemid', how='left') # adding itemid betas
    df_users_items_with_betas = pd.merge(df_users_items_with_betas,df_interactions_grouped_by_creatorid_means[['creatorid', 'b_creatorid']],on='creatorid', how='left') # adding creatorid betas
    df_users_items_with_betas['userid_creatorid'] = df_users_items_with_betas['userid_creatorid'].astype(str)
    df_users_items_with_betas = pd.merge(df_users_items_with_betas,df_interactions_grouped_by_userid_creatorid_means[['userid_creatorid', 'b_userid_creatorid']],on='userid_creatorid', how='left') # adding creatorid betas
 
    # replacing missing values of betas with zero
    df_users_items_with_betas[['b_userid', 'b_userid_creatorid', 'b_itemid', 'b_creatorid']] = df_users_items_with_betas[['b_userid', 'b_userid_creatorid', 'b_itemid', 'b_creatorid']].replace(np.nan, 0)

    # replacing missing values of number of interactions with zero
    df_users_items_with_betas['number_of_interactions'] = df_users_items_with_betas['number_of_interactions'].replace(np.nan, 0)

    # predicting 
    df_users_items_with_betas['prediction_by_residuals']=(
        mu+
        df_users_items_with_betas['b_userid']+
        df_users_items_with_betas['b_itemid']+
        df_users_items_with_betas['b_creatorid']+
        df_users_items_with_betas['b_userid_creatorid'])

    # replacing missing values with the mean
    df_users_items_with_betas['prediction_by_residuals'] = df_users_items_with_betas['prediction_by_residuals'].replace(np.nan, mu)

    ### assigning the probabilities of showing each item to the user 
    # determing the weight of the mean (if only one person has viewed the item, the prediction of showing it will be 
    # 50% the mean and 50% the prediction be residual, because information from one person is not very reilable)
    df_users_items_with_betas['weight_of_mean']=1/(df_users_items_with_betas['number_of_interactions']+1) 
    df_users_items_with_betas['prediction_weighted']=(mu*df_users_items_with_betas['weight_of_mean']+
                                    df_users_items_with_betas['prediction_by_residuals']*(1-df_users_items_with_betas['weight_of_mean']))

    return(df_users_items_with_betas)

########################################################################################
#### A function that calculates recommendation probabilities for one online visitor ####
########################################################################################

def function_calculate_recommendation_probabilities_one_visitor(df_items_with_betas, theme='all', mood='all'):
   
   # filtering the items list by the theme or mood that the visitor has chosen
   df_items_with_betas_filtered=function_df_filter_theme_mood(df_items_with_betas, theme, mood)
   if (len(df_items_with_betas_filtered)==0): df_items_with_betas_filtered=df_items_with_betas # if the visitor viewed all poems that fit their search - they start all over again
   
   # replacing negative values or zeros with 0.0001
   boolean_location=df_items_with_betas_filtered['prediction_weighted']<=0
   df_items_with_betas_filtered=df_items_with_betas_filtered.copy()
   df_items_with_betas_filtered.loc[boolean_location, 'prediction_weighted'] = 0.0001
   
   # calculating the sum of the probabilities
   sum_of_weighted_probabilities=df_items_with_betas_filtered['prediction_weighted'].sum()
   
   # dividing each probability by the sum of weighted probabilities
   df_items_with_betas_filtered=df_items_with_betas_filtered.copy()
   df_items_with_betas_filtered['final_prediction']=df_items_with_betas_filtered['prediction_weighted']/sum_of_weighted_probabilities
   
   # renaming the dataframe
   df_with_final_predictions=df_items_with_betas_filtered
   
   return(df_with_final_predictions)

##############################################################################
#### A function that calculates recommendation probabilities for one user ####
##############################################################################

def function_calculate_recommendation_probabilities_one_user(df_users_requesting_recommendation, df_users_items_with_betas, theme='all', mood='all'):

   ### filtering df_users_items_with_betas so that it will only contain the users requesting recommendations
   # filter the users
   df_users_items_with_betas_one_user = df_users_items_with_betas[df_users_items_with_betas['userid'].isin(df_users_requesting_recommendation['userid'])]

   # removing items that the user viewed in the past
   df_users_items_with_betas_one_user=df_users_items_with_betas_one_user.copy()
   df_users_items_with_betas_one_user=function_removing_viewed_items(df_interactions, df_users_items_with_betas_one_user)
   
   # switching to the visitor function in case its a new user
   if len(df_users_items_with_betas_one_user)==0: # in this case the user is new
       df_items_with_betas = function_calculate_probabilities_visitors(df_interactions, df_items) # creating the betas
       df_with_final_predictions = function_calculate_recommendation_probabilities_one_visitor(df_items_with_betas, theme, mood) # calculating the final predictions
   
   else:

       # filtering the items list by the theme or mood that the user has chosen
       df_with_final_predictions=function_df_filter_theme_mood(df_users_items_with_betas_one_user, theme, mood)
       if (len(df_with_final_predictions)==0): df_with_final_predictions=df_users_items_with_betas_one_user # if the user viewed all poems that fit their search - they start all over again

       # replacing negative values or zeros with 0.0001
       boolean_location=df_with_final_predictions['prediction_weighted']<=0
       df_with_final_predictions=df_with_final_predictions.copy()
       df_with_final_predictions.loc[boolean_location, 'prediction_weighted'] = 0.0001

   return(df_with_final_predictions)

###########################################################################################
###### A function that chooses an item to show a visitor or a user, after            ######
###### the previous function has added the show probabilities.                       ######
###### It chooses one out of two recommendation methods, randomly:                   ######
###### completely random recommendation or based on the 'unique residuals' model     ######
###########################################################################################

def function_choose_one_item_to_display(df_with_final_predictions):
   
  # randomly choosing a recommendation model 
  chosen_model=random.randint(1, 2)
  if (chosen_model==1):
      modelid = 1
      model_name = "Random"
      chosen_item = choice(df_with_final_predictions['itemid'], 1)
  else:
      modelid = 2
      model_name = "Unique_residuals"
      chosen_item = random.choices(df_with_final_predictions['itemid'].values.tolist(), weights=df_with_final_predictions['prediction_weighted'], k=1)
      chosen_item=chosen_item[0] # converting to integer    
   
  # creating a dataframe with the output values
  results={'modelid': [modelid],
           'model_name': [model_name],
           'recommended_item': int(chosen_item)}
  df_results = pd.DataFrame(results)

  return(df_results)

########################################################################################
#### A function that calculates recommendation probabilities for many offline users ####
########################################################################################

"""
This function will be used for users who chose to receive recommendations
via push notifications, email or both. It will run once a day and then the PWA will
send recommended poems to these users.
"""

def function_choose_items_to_display_for_multiple_users(df_users_items_with_betas, df_users_requesting_recommendation):
    
    # creating a dataframe for storing the recommendations 
    df_recommendations=pd.DataFrame(columns=['userid','modelid','model_name','recommended_item'])
    
    ### filtering df_users_items_with_betas according to the user's taste    
    # cacluating the number of days that passed since January first, 2020. This will be used for the 'alternate between fitting and expanding' option     
    # today = date.today() #Today's date
    # past_date = datetime.date(2020, 1, 1) #Jan 1 1970
    # days_since_2020=(today - past_date).days
    # even_day = (days_since_2020 % 2) == 0
    for userid in df_users_requesting_recommendation['userid']:
  
        # filtering df_users_items_with_betas so it will contain only one user
        df_users_items_with_betas_one_user = df_users_items_with_betas.loc[df_users_items_with_betas['userid']==userid]
    
        ### generating the recommendation
        final_predictions_for_one_user=function_calculate_recommendation_probabilities_one_user(df_users_requesting_recommendation, df_users_items_with_betas_one_user, theme='all', mood='all')
        df_recommendation_for_one_user=function_choose_one_item_to_display(final_predictions_for_one_user)  
        
        # adding the userid
        df_recommendation_for_one_user['userid']=userid
        
        # adding the results to the recommendations dataframe
        df_recommendations=pd.concat([df_recommendations, df_recommendation_for_one_user])
    
    return(df_recommendations)