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
from utils.image_utils import analyse_image_func

# Select themes
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']
external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = [dbc.themes.COSMO]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])


CORA_LOGO  = "assets/logo/logo_1.PNG"

# model divs in navigation bar

modal_div_about_us = html.Div(
    [
        dbc.Button("About us", id="open_modal_btn_in_about_us_modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("About us"),
                dbc.ModalBody("This app was developed by the programmers"),
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
                dbc.ModalHeader("Instructions"),
                dbc.ModalBody(
                    children=[
                        html.H6(children='1 . Select Model'),
                        html.P(),
                        html.H6(children='2. Upload image'),
                        html.P(),
                        html.H6(children='3 . Download image'),

                    ]
                ),
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


dropdown =  html.Span(
    id="drop-down-div",
children=[
    html.Span(" Model :  ",id="drop-down-title"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Simple segnet', 'value': '1'},
            {'label': 'segnet_512', 'value': '2'},
            {'label': 'segnet_256', 'value': '3'}
            ],
            value='1',
    )
    ]
)

upload_btn =  dcc.Upload(
        id='upload-image',
        children=html.Div([
            #'Select image ',
            #html.A('Select image Files')
            dbc.Button("Upload image",id="upload-btn",color= "warning")
        ]),
        # Allow multiple files to be uploaded
        multiple=True
    )

subnav_bar = html.Div(
        id="sub-nav-bar",
        children=[dropdown,
        upload_btn],

)

body = dbc.Container([
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



# ********   call back for modal *****************
@app.callback(
    Output("modal_about_us", "is_open"),
    [Input("open_modal_btn_in_about_us_modal", "n_clicks"), Input("close_modal_btn_in_about_us_modal", "n_clicks")],
    [State("modal_about_us", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

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