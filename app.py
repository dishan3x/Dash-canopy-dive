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
external_stylesheets = [dbc.themes.COSMO]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

""" 
navbar = dbc.NavbarSimple(
    brand="Cora",
    color="primary",
    dark=True,
)  """

CORA_LOGO  = "assets/logo_1.PNG"

modal_div_about_us = html.Div(
    [
        dbc.Button("About us", id="open_modal_btn_in_about_us_modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("About us"),
                dbc.ModalBody("This is the content of the modal"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_modal_btn_in_about_us_modal", className="ml-auto")
                ),
            ],
            id="modal_about_us",
        ),
    ]
)

modal_div_instructions= html.Div(
    [
        dbc.Button("Instructions", id="open_modal_btn_in_instructions_modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("Header"),
                dbc.ModalBody("This is the content of the modal"),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close_modal_btn_in_instructions_modal", className="ml-auto")
                ),
            ],
            id="modal_instructions",
        ),
    ]
) 


nav_items = dbc.Row(
    [
        dbc.Col(modal_div_about_us),
        dbc.Col(modal_div_instructions),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(id="logo_img",src=CORA_LOGO, height="30px")),
                    dbc.Col(dbc.NavbarBrand("Cora", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://github.com/dishan3x/Dash-canopy-dive",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(nav_items, id="navbar-collapse", navbar=True),
    ],
    color="dark",
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
            {'label': 'Simple segnet', 'value': '1'},
            {'label': 'Unet', 'value': '2'},
            {'label': 'Arriving soon', 'value': '3'}
        ],
        value='1',
    )

subnav_bar = html.Div(
    id="sub-nav-bar",
    children=[dropdown],
)
body = dbc.Container([
    dcc.Store(id='select_model_value',storage_type='local'),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select image Files')
        ]),
        # Allow multiple files to be uploaded
        multiple=True
    ),

    html.Div(id='output-image-upload'),
])

# Collecting all components
app.layout = html.Div([navbar,subnav_bar,body])

    
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

@app.callback(
    Output("modal_about_us", "is_open"),
    [Input("open_modal_btn_in_about_us_modal", "n_clicks"), Input("close_modal_btn_in_about_us_modal", "n_clicks")],
    [State("modal_about_us", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# ********   call back for modal *****************
@app.callback(
    Output("modal_instructions", "is_open"),
    [Input("open_modal_btn_in_instructions_modal", "n_clicks"), Input("close_modal_btn_in_instructions_modal", "n_clicks")],
    [State("modal_instructions", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


# add callback for toggling the collapse on small screens
@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")],
)
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open




if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server()