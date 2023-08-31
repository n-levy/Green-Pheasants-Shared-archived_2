### create df_users_requesting_recommendation ###

# Get unique values
unique_values = df_users['userid'].unique()

# Create a subset. In this case, we'll take the first 1 unique values.
# Adjust this to get the subset size you want.
subset = unique_values[:5]

# Create new series from subset of unique values
df_users_requesting_recommendation = pd.Series(subset)

# convert the series to a dataframe
df_users_requesting_recommendation = df_users_requesting_recommendation.to_frame()

# change column name to 'userid'
df_users_requesting_recommendation.columns = ['userid']


