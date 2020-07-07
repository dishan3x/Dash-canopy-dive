import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import onnxruntime as rt
from PIL import Image
#from skimage import io
import base64
from io import BytesIO,StringIO
#import plotly.express as px
import dash_bootstrap_components as dbc
from image_utils import analyse_image_func


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


navbar = dbc.NavbarSimple(
    brand="Cora",
    color="primary",
    dark=True,
) 

#dropdown = dbc.DropdownMenu(
#    label="models",
#    className="select-model",
#    children=[
#        dbc.DropdownMenuItem("dishan_segnet_v2"),
#        dbc.DropdownMenuItem("Item 2"),
#        dbc.DropdownMenuItem("Item 3"),
#    ],
#    id="select-model"
#)

dropdown =  dcc.Dropdown(
    id='model-dropdown',
        options=[
            {'label': 'Simple_light_segnet', 'value': '1'},
            {'label': 'Segnet', 'value': '2'},
            {'label': 'Arriving soon', 'value': '3'}
        ],
        value='1',
    )

body = dbc.Container([
    dcc.Store(id='select_model_value',storage_type='local'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])

# Collecting all components
app.layout = html.Div([navbar,dropdown,body])

    
# call back for upload input 
@app.callback(Output(component_id='output-image-upload',component_property='children'),
    [Input(component_id='upload-image', component_property= 'contents')],
    [State('upload-image', 'filename'),
    State('upload-image', 'last_modified'),
    State('model-dropdown','value')])
def update_output(list_of_contents, list_of_names, list_of_dates,model):

    if list_of_contents is not None:
        analysed_information = [
            analyse_image_func(c, n, d, m) for c, n, d, m in
            zip(list_of_contents, list_of_names, list_of_dates,model)]
        return analysed_information
    else:
        return '' # Place holder for the call back

#@app.callback(
#    Output('select_model_value', 'data'),
#    [Input('model-dropdown', 'value')])
#def update_output(value):
#    return value




if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server()