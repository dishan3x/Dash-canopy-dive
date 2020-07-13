import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output


external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = [dbc.themes.COSMO]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


dropdown = dbc.DropdownMenu(
    label="Menu",
    children=[
        dbc.DropdownMenuItem("Item 1", className = "dropdown-button",key="1"),
        dbc.DropdownMenuItem("Item 2", className  = "dropdown-button",key="2" ),
        dbc.DropdownMenuItem("Item 3", className  = "dropdown-button",key="3"),
    ]
)



#dropdown =  html.Span(
#    id="drop-down-div",
#children=[
#    html.Span(" Model :  ",id="drop-down-title"),
#    dcc.Dropdown(
#        id='model-dropdown',
#        options=[
#            {'label': 'Simple segnet', 'value': '1'},
#            {'label': 'segnet_512', 'value': '2'},
#            {'label': 'segnet_256', 'value': '3'}
#            ],
#            value='1',
#    )
#    ]
#)


app.layout =  html.Div([dropdown,html.P(id="item-clicks", className="mt-3")])

@app.callback(
    Output("item-clicks", "children"), [Input(component_className="dropdown-button", component_property="key")]
)
def count_clicks(some):
    print(some)
    if some:
        return f"Button clicked {some} times."
    return "Button not clicked yet."



if __name__ == '__main__':
    app.run_server(debug=True)