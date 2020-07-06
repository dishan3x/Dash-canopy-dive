import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc 

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.A(
        id="some_id",
        children="rhino",     
    ),
    dcc.Store(id='dishan_values',storage_type='local'),
    dcc.Dropdown(
        id='demo-dropdown',
        options=[
            {'label': 'New York City', 'value': 'NYC'},
            {'label': 'Montreal', 'value': 'MTL'},
            {'label': 'San Francisco', 'value': 'SF'}
        ],
        value='NYC'
    ),
    #dbc.Button("Primary",id="down-load-btn2",color="primary"),
    html.Button('click', id='down-load-btn',value="hey_this_is"),

    html.Div(id='dd-output-container')
])

# storing the data in the dccstore and test if can retrieve
@app.callback(
    dash.dependencies.Output('dishan_values', 'data'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    dash.dependencies.Output('down-load-btn', 'value'),
    [dash.dependencies.Input('demo-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    app.run_server(debug=True)