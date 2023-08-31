###################################################
##### Choosing an item for many offline users #####
###################################################

# calculate the betas 
df_users_items_with_betas = function_calculate_probabilities_users(df_users, df_items, df_interactions)

# calculate the probabilities and choose an item for each user, then show all choices in a dataframe
df_results = function_choose_items_to_display_for_multiple_users(df_users_items_with_betas, df_users_requesting_recommendation)

# Notes
"""
1. Run this script right after a scheduled run of the "training_users" script.

2. df_users_items_with_betas and df_interactions should be created by the "training_users" algorithm.

3. df_users, df_items, and df_users_requesting_recommendation should be imported from the PWA 

4. If you would like to create a df_users_requesting_recommendation for testing purposes, 
you can use the script in the Github file "create_df_users_requesting_recommendation_for_testing"

5. If you wish to read a more detailed description of each functions, see 'functions' script.
"""
