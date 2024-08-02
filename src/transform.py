import pandas as pd
import numpy as np

def get_recipe_nutrients(rows,columns,servings,nutrient_db,daily_limits):
    """calculate the nutrients of a recipe based on the ingredients and their amounts"""
    # create dataframe from data table recipe input
    recipe_df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    # merge with nutrient data base
    recipe_nutrients = pd.merge(recipe_df, nutrient_db, how = 'left', left_on = 'Ingredient', right_on = 'name')
    # determine numeric columns and drop amount and id
    numeric_columns = recipe_nutrients.select_dtypes(include='number').columns
    numeric_columns = numeric_columns.drop(['ID','Amount']).to_list()
    # multiply nutrients with multiples of 100g
    recipe_nutrients[numeric_columns] =  recipe_nutrients[numeric_columns].multiply(recipe_nutrients["Amount"]/100, axis="index")
    # get the total nutrients of recipe per serving
    total_nutrients = recipe_nutrients.sum(axis=0, numeric_only=True).to_frame().T
    #  get the nutrients for one serving
    serving_nutrients = total_nutrients.multiply(1/servings)
    # create copy of recipe nutrients ingredientwise nutrients
    ingredient_nutrients_serving = recipe_nutrients.copy()
     #  get the ingredientwise nutrients for one serving
    ingredient_nutrients_serving[numeric_columns] = ingredient_nutrients_serving[numeric_columns].multiply(1/servings)
    # drop columns that are not needed
    ingredient_nutrients_serving.drop(columns = ['Amount','ID','name','Food Group'],inplace=True)
    # set index to ingredient
    ingredient_nutrients_serving.set_index('Ingredient',inplace=True)
    # group by ingredient and sum
    ingredient_nutrients_serving = ingredient_nutrients_serving.astype('float32')
    ingredient_nutrients_serving = ingredient_nutrients_serving.groupby(ingredient_nutrients_serving.index).sum()
    # get the serving summary (transposed view of ingredient_nutrients_serving)
    serving_summary = ingredient_nutrients_serving.T.merge(daily_limits,how='inner',left_index=True,right_on='Nutrient').set_index('Nutrient')
    # reorder rows
    serving_summary = serving_summary.loc[daily_limits['Nutrient'],:]
    # move last column to first column
    cols = serving_summary.columns.tolist()
    serving_summary = serving_summary.astype('float32')
    cols = cols[-1:] + cols[:-1]
    serving_summary = serving_summary[cols]
    # calculate relative amout of daily value
    serving_summary[cols[1:]] =  serving_summary[cols[1:]].multiply(1/serving_summary['Daily value'], axis="index")
    # calculate total amount of nutrients
    serving_summary['Serving Total'] = serving_summary[cols[1:]].sum(axis=1)
    #reorder columns (move total to the left)
    serving_summary = serving_summary.reset_index()
    cols = serving_summary.columns.tolist()
    cols = cols[:2] + [cols[-1]] + cols[2:-1]
    serving_summary = serving_summary[cols]
    
    return ingredient_nutrients_serving, serving_nutrients, serving_summary

# Daily macros of interest
daily_macros = ['Calories (kcal)','Fat (g)','Protein (g)','Carbohydrate (g)','Added Sugar (g)','Sodium (mg)','Cholesterol (mg)','Fiber (g)']
# Caloric multiplier for Fat, Protein and Carbohydrates
caloric_multiplier = np.array([9,4,4])