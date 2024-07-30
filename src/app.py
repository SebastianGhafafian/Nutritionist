from dash import Dash, dash_table, html, dcc, Input, Output, State, ctx
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from scipy.stats import beta
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
import json

### todo list
# figure out data manipulation depending on callbacks


def highlight_values(df):
    

    conditional_format_low = [
    {
        'if': {
            'filter_query': '{{{col}}} > 0.05 && {{{col}}} < 0.15'.format(col=col),
            'column_id': col
        },
        'backgroundColor': 'skyblue',
        'color':'white',
        'fontWeight': 'bold'
    }
    for col in df.columns if col not in ['Nutrient','Daily value']]


    conditional_format_high = [
    {
        'if': {
            'filter_query': '{{{col}}} > 0.15'.format(col=col),
            'column_id': col
        },
        'backgroundColor': 'steelblue',
        'color':'white',
        'fontWeight': 'bold'
    }
    for col in df.columns if col not in ['Nutrient','Daily value']]


    format_leftside=[{
            'if': {
                'column_id': 'Nutrient',
            },
            'width':'150px'
            # 'backgroundColor': 'lightgrey'
        },
        {
            'if': {
                'column_id': 'Daily value',
            },
            'minWidth':'60px',
            'maxWidth':'60px'
        }
        ]
    
    return conditional_format_low + conditional_format_high # +format_leftside
def get_recipe_nutrients(rows,columns,servings,nutrient_db):
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
    total_nutrients.multiply(1/servings)
    #  get the nutrients for one serving
    serving_nutrients = total_nutrients.multiply(1/servings)
    ingredient_nutrients_serving = recipe_nutrients.copy()
    ingredient_nutrients_serving[numeric_columns] = ingredient_nutrients_serving[numeric_columns].multiply(1/servings)

    return ingredient_nutrients_serving, total_nutrients, recipe_nutrients, serving_nutrients

# Load daily values
daily_limits = pd.read_csv('../data/Daily_Limits.csv',delimiter=';')

caloric_multiplier = np.array([9,4,4])
ingredient_table_data = pd.DataFrame({'Ingredient':[],'Amount':[]})

daily_marcos = ['Calories (kcal)','Fat (g)','Protein (g)','Carbohydrate (g)','Added Sugar (g)','Sodium (mg)','Cholesterol (mg)','Fiber (g)']
nutrient_database = pd.read_csv('../data/database.csv')



with open('../data/recipes.json', 'r') as openfile:
    # Reading from json file
    recipes = json.load(openfile)


app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB,'.assets/custom.css'])
server = app.server
percentage_format = dash_table.FormatTemplate.percentage(0)
#formatting

# app.config.suppress_callback_exceptions=True

#     fig.update_layout(showlegend=True)


# html components

recipe_dropdown = html.Div(
    [
        dbc.Label("Load recipe", html_for="size"),
        dcc.Dropdown(list(recipes.keys()),
                 id='recipe-dropdown',placeholder='Select a recipe...',),
        html.Button('Load Recipe', id='load-recipe-button', n_clicks=0)
    ],
    # style={"width": "50%"},
    className="mt-2",
)



ingredient_dropdown = html.Div(
    [
        dbc.Label("Ingredient", html_for="size"),
        dcc.Dropdown(nutrient_database.name,
                 id='ingredient-dropdown')
    ],
    className="mt-2",
)

ingredient_grams_input = html.Div(
    [
        dbc.Label("Amount in grams", html_for="size"),
        dcc.Input(id='ingredient-grams-input', type='number',value = 100, min=1, max=5000, step=1)
    ],
    className="mt-2",
)

column_names = ingredient_table_data.columns
ingredient_table = dash_table.DataTable(
        id='ingredient-table',
        data = ingredient_table_data.to_dict('records'), columns = [{"name": column_names[0], "id": column_names[0]},
                                                 {"name": column_names[1], "id": column_names[1],'type':'numeric'},],
        editable=True,
        row_deletable = True,
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            }
    )

servings_input = html.Div(
    [
        dbc.Label("Number of servings", html_for="size"),
        dcc.Input(id='servings-input', type='number',value = 2, min=1, max=100, step=1)
    ],
    className="mt-2",
)

recipe_name_input = dcc.Input(
            id="recipe-name-input",
            type='text',
            placeholder='enter recipe name',
        )

heading = html.H4(
    "DietPy", className="bg-primary text-white p-2"
)
micro_summary_table = html.Div(id='micro-summary-table')

# define panels
control_panel = dbc.Card(
    dbc.CardBody(
        [recipe_dropdown,
         ingredient_dropdown,
         ingredient_grams_input,
         html.Button('Add Row', id='editing-rows-button', n_clicks=0),
         servings_input,
         ingredient_table,
         recipe_name_input,
         html.Button('Save recipe', id='save-recipe-button', n_clicks=0)],
        className="bg-light",
    )
)
graph = dbc.Card(
    [dcc.Tabs([
        dcc.Tab(label='Macronutrients', children=[
                    html.Div(id="error_msg", className="text-danger"), dcc.Graph(id="pie-graph"),
                    html.Div(id="error2_msg", className="text-danger"), dcc.Graph(id="daily-macro-bar-graph")
        ]),
        dcc.Tab(label='Detail View', children=[
            micro_summary_table
        ]),

    ]) 

    ]
)

# define layout
app.layout = html.Div(
    [heading, dbc.Row([dbc.Col(control_panel, md=3), dbc.Col(graph, md=9)])]
)


'''Callbacks'''
# table update

@app.callback(
        Output('recipe-dropdown','options'),
        Input('save-recipe-button','n_clicks'),
        State('ingredient-table','data'),
        State('recipe-name-input','value'),
)
def export_recipe(n_clicks,data,recipe_name):
    # turn data table into a dict
    if n_clicks > 0:
        # update recipes with new recipe
        recipes.update({recipe_name:data})
        # update recipes.json file
        with open("./data/recipes.json", "w") as outfile:
            json.dump(recipes, outfile,indent=4, separators=(',', ': '))
        # return new list for recipe drop down menu
        return list(recipes.keys())
    else:
        raise PreventUpdate

@app.callback(
    Output('ingredient-table', 'data'),
    Input('load-recipe-button', 'n_clicks'),
    Input('editing-rows-button', 'n_clicks'),
    State('recipe-dropdown', 'value'),
    State('ingredient-dropdown', 'value'),
    State('ingredient-grams-input', 'value'),
    State('ingredient-table', 'data'),
    State('ingredient-table', 'columns'))
def update_table(n_clicks_recipe,n_clicks_ingredient,recipe,ingredient,grams, rows, columns):
    # perfrom update of table when recipe is loaded
    if "load-recipe-button" == ctx.triggered_id:
        used_recipe = recipes[recipe]
        # loop over the recipe ingredients
        # fill will data table with corresponding ingred and grams
        rows = used_recipe
        # rows = [{'Ingredient': key, 'Amount': value} for key, value in used_recipe.items()]
    # perfrom update of table when ingredient is added
    elif 'editing-rows-button' == ctx.triggered_id:
        # print('ingred')
        rows.append({c['id']: i for c,i in zip(columns,[ingredient,grams])})


    return rows



#create callback for pie chart
@app.callback(
    Output(component_id="pie-graph",component_property='figure'),
    [Input('ingredient-table', 'data'),
    Input('ingredient-table', 'columns'),
    Input('servings-input', 'value')
])
def pie_graph(rows,columns,servings):
    if len(rows) == 0:
        raise PreventUpdate
    else:


        ingredient_nutrients_serving, total_nutrients, recipe_nutrients, serving_nutrients = get_recipe_nutrients(rows, columns, servings, nutrient_database)
        #plotdata
        macros = ['Fat (g)','Protein (g)','Carbohydrate (g)']
        macro_plot_names = [name.split('(')[0] for name in macros]
        macro_data = serving_nutrients[macros].T.to_numpy().reshape(-1)
        #plot templates
        hovertemplate = "<b>%{label}</b><br>%{percent}<br>"
        texttemplate_grams = "<b>%{label}</b><br>%{value:.0f} grams<br>"
        texttemplate_kcal = "<b>%{label}</b><br>%{value:.0f} kcal<br>"
        colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

        # piechart
        fig = make_subplots(rows=1,
                            cols=2,
                            specs=[[{"type": "domain"}, {"type": "domain"}]],
                            subplot_titles=[f'Weight proportion (Total/Serving: {macro_data.sum():.0f} g)',
                                            f'Caloric proportions (Total/Serving: {(macro_data* caloric_multiplier).sum():.0f} kcal)'])
        # Pie chart in grams
        fig.add_trace(trace = go.Pie(values=macro_data,labels = macro_plot_names,hovertemplate=hovertemplate,texttemplate= texttemplate_grams),
                row=1, col=1)
        #Pie chart in kcal
        fig.add_trace(trace = go.Pie(values=macro_data * caloric_multiplier, labels = macro_plot_names,hovertemplate=hovertemplate,texttemplate= texttemplate_kcal),
                row=1, col=2)
        fig.update_traces(hoverinfo='label+percent',
                        marker=dict(colors=colors, line=dict(color='#000000', width=2)))
        fig.update_layout(title_text='Macronutrients per serving')

        return fig



@app.callback(
    Output(component_id="daily-macro-bar-graph",component_property='figure'),
    [
    Input('ingredient-table', 'data'),
    Input('ingredient-table', 'columns'),
    Input('servings-input', 'value')
])
def makro_bar_graph(rows,columns,servings):
    if len(rows) == 0:
        raise PreventUpdate
    else:
        ingredient_nutrients_serving, total_nutrients, recipe_nutrients, serving_nutrients = get_recipe_nutrients(rows, columns, servings, nutrient_database)

        # transpose data for bar plot
        plot_df = serving_nutrients[daily_marcos].T.reset_index().set_axis(['Nutrient','Amount per Serving'], axis = 1)
        # add daily limits
        plot_df = pd.merge(plot_df, daily_limits, on = 'Nutrient' , how='inner')
        # calculate percentages reaches of daily limit
        plot_df['Percent of Limit'] = plot_df['Amount per Serving'] / plot_df['Daily value'] * 100 
        # remove grams from nutrient name
        plot_df['Nutrient'] = plot_df.apply(lambda x: x['Nutrient'].split('(')[0], axis = 1)
        # Bar plot
        fig = go.Figure(
            data=[
                # daily value (100 %)
                go.Bar(x= 100 * np.ones_like(plot_df['Nutrient']),y = plot_df['Nutrient'],orientation= 'h',marker=dict(opacity = 1),name='Daily value'),
                # Percentage of daily limit reached 
                go.Bar(x=plot_df['Percent of Limit'],y = plot_df['Nutrient'],orientation= 'h', marker=dict(opacity = 0.6),name='Contribution to Daily value'),
                ]
            )
        fig.update_layout(barmode='overlay',title_text = 'Contribution of one serving to daily values',xaxis_title='Amount per serving in %',)

        return fig


@app.callback(
    Output(component_id="micro-summary-table",component_property='children'),
    [
    Input('ingredient-table', 'data'),
    Input('ingredient-table', 'columns'),
    Input('servings-input', 'value')
])
def micro_summary_table(rows,columns,servings):
    if len(rows) == 0:
        raise PreventUpdate
    else:


        # ingredient_nutrients_serving, total_nutrients, recipe_nutrients, serving_nutrients = get_recipe_nutrients(rows, columns, servings, nutrient_database)
        # ingredient_nutrients_serving.set_index('Ingredient',inplace=True)
        # print(ingredient_nutrients_serving)
        # ingredient_nutrients_serving = ingredient_nutrients_serving.groupby(ingredient_nutrients_serving.index).sum().reset_index()
        # print(ingredient_nutrients_serving)
        # # print(daily_limits)
        # serving_summary = ingredient_nutrients_serving.T.merge(daily_limits,how='inner',left_index=True,right_on='Nutrient').set_index('Nutrient')






        ingredient_nutrients_serving, total_nutrients, recipe_nutrients, serving_nutrients = get_recipe_nutrients(rows, columns, servings, nutrient_database)
        ingredient_nutrients_serving.drop(columns = ['Amount','ID','name','Food Group'],inplace=True)
        print(ingredient_nutrients_serving)
        ingredient_nutrients_serving.set_index('Ingredient',inplace=True)
        ingredient_nutrients_serving = ingredient_nutrients_serving.groupby(ingredient_nutrients_serving.index).sum()
        print(ingredient_nutrients_serving)
        # print(daily_limits)
        serving_summary = ingredient_nutrients_serving.T.merge(daily_limits,how='inner',left_index=True,right_on='Nutrient').set_index('Nutrient')
        serving_summary = serving_summary.loc[daily_limits['Nutrient'],:]
        print(serving_summary)
        # serving_summary = serving_summary.drop(['Calories','Fat (g)', 'Protein (g)', 'Carbohydrate (g)'])
        cols = serving_summary.columns.tolist()
        serving_summary = serving_summary.astype('float32')
        cols = cols[-1:] + cols[:-1]
        serving_summary = serving_summary[cols]

        serving_summary[cols[1:]] =  serving_summary[cols[1:]].multiply(1/serving_summary['Daily value'], axis="index")
        serving_summary['Serving Total'] = serving_summary[cols[1:]].sum(axis=1)
        serving_summary = serving_summary.reset_index()
        cols = serving_summary.columns.tolist()
        # print(cols[:2] ,cols[-1] ,cols[2:-1])
        cols = cols[:2] + [cols[-1]] + cols[2:-1]
        
        serving_summary = serving_summary[cols]
        columns = []
        for i in serving_summary.columns:
            if i == 'Daily value':
                columns.append({"name": i, "id": i,'type':'numeric', 'format': {'specifier': '.2f'}},)
            elif i == 'Nutrient':
                columns.append({"name": i, "id": i})
            else:
                columns.append({"name": i, "id": i,"type":'numeric','format':percentage_format} )
        A = dash_table.DataTable(
            serving_summary.to_dict('records'),
            # [{"name": i, "id": i,"type":'numeric','format':percentage_format}  if i not in ['Daily value','Nutrient'] else {"name": i, "id": i,"type":'numeric'} for i in serving_summary.columns])
            columns=columns,
            fixed_rows={'headers': True},

            style_cell={
                'maxWidth':'50px',
                'minWidth':'50px',
                'whiteSpace': 'normal',
                'height': 'auto',
                # 'overflowX': 'auto'
                # 'textOverflow': 'ellipsis'
            },
            style_table={'minWidth': '100%',"height": "70vh", "maxHeight": "70vh", 'overflowY': 'auto'},
            style_data_conditional=highlight_values(serving_summary),
            style_cell_conditional= [{
            'if': {
                'column_id': 'Nutrient',
            },
            'minWidth':'170px',
            'maxWidth':'170px'
            # 'backgroundColor': 'lightgrey'
        },
        {
            'if': {
                'column_id': 'Daily value',
            },
            'minWidth':'60px',
            'maxWidth':'60px'
        }
        ]
        )
        return A


if __name__ == "__main__":
    app.run_server(debug=True)
