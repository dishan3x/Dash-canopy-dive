import datetime
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import onnxruntime as rt
from PIL import Image
import base64
from io import BytesIO,StringIO
import dash_bootstrap_components as dbc
from utils.image_utils import analyse_image_func
from utils.app_models import app_information,app_instructions
import time


# Select themes
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']
external_stylesheets = [dbc.themes.BOOTSTRAP]
external_stylesheets = [dbc.themes.COSMO]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ])

# logo for the app
CORA_LOGO  = "assets/logo/logo_1.PNG"


##########   Nav bar   ######################

# model component

# Developer infromations modal
modal_div_developer_info = app_information()

# Instruction model
modal_div_instructions  = app_instructions()

# Nav items component
nav_items = dbc.Row(
    [
        dbc.Col(modal_div_developer_info),
        dbc.Col(modal_div_instructions),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)


# Develop sub nav bar ###############################

# Deep learning model select drop down component
dropdown =  html.Span(

    id="drop-down-div",
    children=[
    html.Span(" Model :  ",id="drop-down-title"),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Segnet', 'value': '1'},
            {'label': 'Unet', 'value': '2'},
            {'label': 'segnet_mohomad', 'value': '3'}
            ],
            value='1',
    )
    ]
)

# upload image button component
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

# Create nav bar 
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
        dropdown,
        upload_btn,
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(nav_items, id="navbar-collapse", navbar=True),
    ],
    color="dark",
    dark=True,
)

# Create the sub nav bar
#subnav_bar = html.Div(
#        id="sub-nav-bar",
#        children=[dropdown,
#        upload_btn],

#)

# Body container component  ###########################
body = dbc.Container([
    html.Div(id='output-image-upload'),
])


# Collecting all components for app layout ##########
app.layout = html.Div([navbar,body])


# Call backs ###################################### 

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
        time.sleep(3),
        loading_spinner = html.Div([
            dbc.Spinner(html.Div(id="loading-output")),
            ])
        return '' # Place holder for the call back

# Call back for modals ###########################

@app.callback(
    Output("modal_about_us", "is_open"),
    [Input("open-modal-btn-in-about-us-modal", "n_clicks"), Input("close-modal-btn-in-about-us-modal", "n_clicks")],
    [State("modal_about_us", "is_open")],
)
def toggle_modal(open_btn, close_btn, is_open):
    if open_btn or close_btn:
        return not is_open
    return is_open

@app.callback(
    Output("modal_instructions", "is_open"),
    [Input("open-modal-btn-in-instructions-modal", "n_clicks"), Input("close_modal_btn_in_instructions_modal", "n_clicks")],
    [State("modal_instructions", "is_open")],
)
def toggle_modal(open_btn, close_btn, is_open):
    if open_btn or close_btn:
        return not is_open
    return is_open



# callback for toggling the collapse button on small screens ############

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