from dash import Dash, dash_table, html, dcc, Input, Output, State, ctx
import numpy as np
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
import json
import base64
import formats
import transform


"""Load Data"""
# Daily values
daily_limits = pd.read_csv("../data/Daily_Limits.csv", delimiter=";")
# Nutrient database
nutrient_database = pd.read_csv("../data/database.csv")
# Recipes
with open("../data/recipes.json", encoding="utf-8", mode="r") as openfile:
    recipes = json.load(openfile)  # read from json file


# Initialize ingredient table
ingredient_table_data = pd.DataFrame({"Ingredient": [], "Amount": []})


app = Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB, "../assets/custom.css"])
server = app.server


"""html components"""
upload_field = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drop or ", html.A("Select File")]),
            style={
                "width": "95%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div(id="output-data-upload"),
    ]
)

recipe_dropdown = html.Div(
    [
        dbc.Label("Load recipe", html_for="size"),
        dcc.Dropdown(
            list(recipes.keys()),
            id="recipe-dropdown",
            placeholder="Select a recipe...",
        ),
        html.Button("Load Recipe", id="load-recipe-button", n_clicks=0),
    ],
    # style={"width": "50%"},
    className="mt-2",
)


ingredient_dropdown = html.Div(
    [
        dbc.Label("Ingredient", html_for="size"),
        dcc.Dropdown(nutrient_database.name, id="ingredient-dropdown"),
    ],
    className="mt-2",
)

ingredient_grams_input = html.Div(
    [
        dbc.Label("Amount in grams", html_for="size"),
        dcc.Input(
            id="ingredient-grams-input",
            type="number",
            value=100,
            min=1,
            max=5000,
            step=1,
        ),
    ],
    className="mt-2",
)

column_names = ingredient_table_data.columns
ingredient_table = dash_table.DataTable(
    id="ingredient-table",
    data=ingredient_table_data.to_dict("records"),
    columns=[
        {"name": column_names[0], "id": column_names[0]},
        {"name": column_names[1], "id": column_names[1], "type": "numeric"},
    ],
    editable=True,
    row_deletable=True,
    style_data={
        "whiteSpace": "normal",
        "height": "auto",
    },
)

servings_input = html.Div(
    [
        dbc.Label("Number of servings", html_for="size"),
        dcc.Input(id="servings-input", type="number", value=2, min=1, max=100, step=1),
    ],
    className="mt-2",
)

recipe_name_input = dcc.Input(
    id="recipe-name-input",
    type="text",
    placeholder="enter recipe name",
)

heading = html.H4("Nutritionist", className="bg-primary text-white p-2")
detail_table = html.Div(id="detail-table")

# define panels
# Left panel
control_panel = dbc.Card(
    dbc.CardBody(
        [
            upload_field,
            recipe_dropdown,
            ingredient_dropdown,
            ingredient_grams_input,
            html.Button("Add Row", id="editing-rows-button", n_clicks=0),
            servings_input,
            ingredient_table,
            recipe_name_input,
            html.Button("Save recipe", id="save-recipe-button", n_clicks=0),
            html.Button("Download recipes", id="download-button"),
            dcc.Download(id="download-recipes"),
            # stores
            dcc.Store(id="recipe-store", data=recipes),
            dcc.Store(id="serving-nutrients-store", data=[]),
            dcc.Store(id="ingredient-nutrients-serving-store", data=[]),
            dcc.Store(id="serving-summary-store", data=[]),
        ],
        className="bg-light",
    )
)
# right panel
graph = dbc.Card(
    [
        dcc.Tabs(
            [
                dcc.Tab(
                    label="Macronutrients",
                    children=[
                        html.Div(id="error_msg", className="text-danger"),
                        dcc.Graph(id="pie-graph"),
                        html.Div(id="error2_msg", className="text-danger"),
                        dcc.Graph(id="daily-macro-bar-graph"),
                    ],
                ),
                dcc.Tab(label="Detail View", children=[detail_table]),
            ]
        )
    ]
)

"""Layout"""
app.layout = html.Div(
    [heading, dbc.Row([dbc.Col(control_panel, md=4), dbc.Col(graph, md=8)])]
)


"""Callbacks"""
@app.callback(
    Output(
        component_id="ingredient-nutrients-serving-store", component_property="data"
    ),
    Output(component_id="serving-nutrients-store", component_property="data"),
    Output(component_id="serving-summary-store", component_property="data"),
    [
        Input("ingredient-table", "data"),
        Input("ingredient-table", "columns"),
        Input("servings-input", "value"),
    ],
)
def recipe_nutrients(rows, columns, servings):
    """Calculate the nutrient values of the recipe"""
    if len(rows) == 0:
        raise PreventUpdate
    else:
        ingredient_nutrients_serving, serving_nutrients, serving_summary = (
            transform.get_recipe_nutrients(
                rows, columns, servings, nutrient_database, daily_limits
            )
        )

        return (
            ingredient_nutrients_serving.to_dict(
                "records"
            ),  # store the nutrient values of the ingredients
            serving_nutrients.to_dict(
                "records"
            ),  # store the nutrient values of the recipe
            serving_summary.to_dict(
                "records"
            ),  # store the nutrient values of the recipe reformatted
        )


@app.callback(Output("recipe-dropdown", "options"), Input("recipe-store", "data"))
def update_recipe_dropdown(recipes_dict):
    """Update the recipe dropdown with the recipes in the recipe store"""
    return list(recipes_dict.keys())


@app.callback(
    Output("recipe-store", "data"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    State("recipe-store", "data"),
    Input("save-recipe-button", "n_clicks"),
    State("ingredient-table", "data"),
    State("recipe-name-input", "value"),
)
def update_recipes(contents, filename, recipes_dict, n_clicks, data, recipe_name):
    """Update the recipe store with new recipes"""
    if "save-recipe-button" == ctx.triggered_id:  # if new recipes is saved
        if n_clicks > 0:
            # update recipes with new recipe
            recipes_dict.update({recipe_name: data})

        return recipes_dict

    elif "upload-data" == ctx.triggered_id:  # if recipe file is uploaded
        if contents is None:
            raise PreventUpdate
        if not filename.endswith(".json"):
            return "Please upload a file with the .json extension"

        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        recipes_dict = json.loads(decoded)
        return recipes_dict

    else:
        raise PreventUpdate


@app.callback(
    Output("download-recipes", "data"),
    Input("download-button", "n_clicks"),
    State("recipe-store", "data"),
    prevent_initial_call=True,
)
def download_recipes(n_clicks, recipes_dict):
    """Download recipes as json file"""
    return dict(
        content=json.dumps(recipes_dict, indent=4, separators=(",", ": ")),
        filename="recipes.json",
    )


@app.callback(
    Output("ingredient-table", "data"),
    Input("load-recipe-button", "n_clicks"),
    Input("editing-rows-button", "n_clicks"),
    State("recipe-dropdown", "value"),
    State("ingredient-dropdown", "value"),
    State("ingredient-grams-input", "value"),
    State("ingredient-table", "data"),
    State("ingredient-table", "columns"),
    State("recipe-store", "data"),
)
def update_table(
    n_clicks_recipe,
    n_clicks_ingredient,
    recipe,
    ingredient,
    grams,
    rows,
    columns,
    recipes_dict,
):
    """Update the ingredient table with the ingredients of the loaded recipe
    or with the ingredients added by the user"""
    # perfrom update of table when recipe is loaded
    if "load-recipe-button" == ctx.triggered_id:
        used_recipe = recipes_dict[recipe]
        rows = used_recipe
    # perform update of table when ingredient is added
    elif "editing-rows-button" == ctx.triggered_id:
        rows.append({c["id"]: i for c, i in zip(columns, [ingredient, grams])})

    return rows


# create callback for pie chart
@app.callback(
    Output(component_id="pie-graph", component_property="figure"),
    Input(component_id="serving-nutrients-store", component_property="data"),
)
def pie_graph(serving_nutrients_store):
    """Create tow pie chart with the macronutrient values of one serving of the recipe"""
    if len(serving_nutrients_store) == 0:
        raise PreventUpdate
    else:
        serving_nutrients = pd.DataFrame(serving_nutrients_store)
        # plotdata
        macros = ["Fat (g)", "Protein (g)", "Carbohydrate (g)"]
        macro_plot_names = [name.split("(")[0] for name in macros]
        macro_data = serving_nutrients[macros].T.to_numpy().reshape(-1)
        # plot templates
        hovertemplate = "<b>%{label}</b><br>%{percent}<br>"
        texttemplate_grams = "<b>%{label}</b><br>%{value:.0f} grams<br>"
        texttemplate_kcal = "<b>%{label}</b><br>%{value:.0f} kcal<br>"
        colors = ["gold", "mediumturquoise", "darkorange", "lightgreen"]

        # piechart
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=[
                "Weight proportion",
                f"Caloric proportions (Total/Serving: {(macro_data * transform.caloric_multiplier).sum():.0f} kcal)",
            ],
        )
        # Pie chart in grams
        fig.add_trace(
            trace=go.Pie(
                values=macro_data,
                labels=macro_plot_names,
                hovertemplate=hovertemplate,
                texttemplate=texttemplate_grams,
                insidetextorientation = "horizontal",
            ),
            row=1,
            col=1,
        )
        # Pie chart in kcal
        fig.add_trace(
            trace=go.Pie(
                values=macro_data * transform.caloric_multiplier,
                labels=macro_plot_names,
                hovertemplate=hovertemplate,
                texttemplate=texttemplate_kcal,
                insidetextorientation = "horizontal",
            ),
            row=1,
            col=2,
        )
        fig.update_traces(
            hoverinfo="label+percent",
            marker=dict(colors=colors, line=dict(color="#000000", width=2)),

        )
        fig.update_layout(
            title_text="Macronutrients per serving",
            
        )
        fig.layout.annotations[0].update(y=-0.2)
        fig.layout.annotations[1].update(y=-0.2)

        return fig


@app.callback(
    Output(component_id="daily-macro-bar-graph", component_property="figure"),
    Input(component_id="serving-nutrients-store", component_property="data"),
)
def makro_bar_graph(serving_nutrients_store):
    """Create a bar plot with the contribution of one serving to the daily values"""
    if len(serving_nutrients_store) == 0:
        raise PreventUpdate
    else:
        serving_nutrients = pd.DataFrame(serving_nutrients_store)

        # transpose data for bar plot
        plot_df = (
            serving_nutrients[transform.daily_macros]
            .T.reset_index()
            .set_axis(["Nutrient", "Amount per Serving"], axis=1)
        )
        # add daily limits
        plot_df = pd.merge(plot_df, daily_limits, on="Nutrient", how="inner")
        # calculate percentages reaches of daily limit
        plot_df["Percent of Limit"] = (
            plot_df["Amount per Serving"] / plot_df["Daily value"] * 100
        )
        # remove grams from nutrient name
        plot_df["Nutrient"] = plot_df.apply(
            lambda x: x["Nutrient"].split("(")[0], axis=1
        )
        # Bar plot
        fig = go.Figure(
            data=[
                # daily value (100 %)
                go.Bar(
                    x=100 * np.ones_like(plot_df["Nutrient"]),
                    y=plot_df["Nutrient"],
                    orientation="h",
                    marker=dict(opacity=1),
                    name="Daily value",
                ),
                # Percentage of daily limit reached
                go.Bar(
                    x=plot_df["Percent of Limit"],
                    y=plot_df["Nutrient"],
                    orientation="h",
                    marker=dict(opacity=0.6),
                    name="Contribution to Daily value",
                ),
            ]
        )
        fig.update_layout(
            barmode="overlay",
            title_text="Contribution of one serving to daily values",
            xaxis_title="Amount per serving in %",
        )

        return fig


@app.callback(
    Output(component_id="detail-table", component_property="children"),
    Input(component_id="serving-summary-store", component_property="data"),
)
def get_detail_table(serving_summary_store):
    """Create a detailed table with the nutrient values of the recipe"""
    if len(serving_summary_store) == 0:
        raise PreventUpdate
    else:
        serving_summary = pd.DataFrame(serving_summary_store)
        columns = []
        for i in serving_summary.columns:
            if i == "Daily value":
                columns.append(
                    {
                        "name": i,
                        "id": i,
                        "type": "numeric",
                        "format": {"specifier": ".2f"},
                    },
                )
            elif i == "Nutrient":
                columns.append({"name": i, "id": i})
            else:
                columns.append(
                    {"name": i,
                     "id": i,
                     "type": "numeric",
                     "format": dash_table.FormatTemplate.percentage(0)}
                )
        summary_table = dash_table.DataTable(
            serving_summary.to_dict("records"),
            columns=columns,
            fixed_rows={"headers": True},
            style_cell={
                "maxWidth": "50px",
                "minWidth": "50px",
                "whiteSpace": "normal",
                "height": "auto",
            },
            style_table={
                "minWidth": "100%",
                "height": "70vh",
                "maxHeight": "70vh",
                "overflowY": "auto",
            },
            style_data_conditional=formats.highlight_values(serving_summary),
            style_cell_conditional=[
                {
                    "if": {
                        "column_id": "Nutrient",
                    },
                    "minWidth": "170px",
                    "maxWidth": "170px",
                },
                {
                    "if": {
                        "column_id": "Daily value",
                    },
                    "minWidth": "60px",
                    "maxWidth": "60px",
                },
            ],
        )
        return summary_table


if __name__ == "__main__":
    app.run_server(debug=True)
