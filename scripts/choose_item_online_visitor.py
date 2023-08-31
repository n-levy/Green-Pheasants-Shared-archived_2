##################################################
##### Choosing an item for an online visitor #####
##################################################

# Filter the interactions dataframe, so that it would include
# only items with the theme or with the mood that the user chose.
df_interactions = function_df_filter_theme_mood(df_interactions, theme='all', mood='all')

# calculate the probabilities
df_with_final_predictions = function_calculate_recommendation_probabilities_one_visitor(df_items_with_betas, theme='all', mood='all') 

# choose one item to display
df_results = function_choose_one_item_to_display(df_with_final_predictions)

# Notes
"""
1. df_interactions and df_items_with_betas should be created by the "training_users" algorithm.

2. df_users, df_items, and df_users_requesting_recommendation should be imported from the PWA 

3. If you wish to read a more detailed description of each functions, see 'functions' script.
"""