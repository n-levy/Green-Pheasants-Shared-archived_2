#####################################
##### Train the model for users #####
#####################################

""" Prepare the data for analysis """
### Prepare the interactions dataframe
# Remove rows with missing values in columns needed for the recommendation code
df_interactions = function_remove_rows_with_missing_values(df_interactions)

# Create necessary cloumns in df_interactions
df_interactions = function_add_columns(df_interactions)

# Create a dataframe with all combinations of users and items
df_users_items = function_attach_userids_to_items(df_users, df_items)

# Remove items that the users have viewed in the past
df_users_items = function_removing_viewed_items(df_interactions, df_users_items)

""" Calculate the betas """
# calculate the betas 
df_users_items_with_betas = function_calculate_probabilities_users(df_users, df_items, df_interactions)

# Note
"""
The input datasets should be imported from the PWA's database (df_users, df_items, df_interactions).

"""


