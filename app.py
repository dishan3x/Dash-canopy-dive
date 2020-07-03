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
import plotly.express as px
import dash_bootstrap_components as dbc

# Newly added to testing the iamge download
from urllib.parse import quote as urlquote
from flask import Flask, send_from_directory


#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


navbar = dbc.NavbarSimple(
    brand="Cora",
    color="primary",
    dark=True,
) 

body =dbc.Container([
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
    html.A(
        id="download-content",
        download="image.png",
        children=[
            dbc.Button(
                "Download image",
                id="down-load_button",
                color="primary",
                #children=html.Img(
                #    src="assets/some_image.svg",
                #    className="button-image",
                #),
                className="inline_button",
            )
        ],
    )
])

# Collecting all components
app.layout = html.Div([navbar,body])

def parse_contents(contents, filename, date, contructed_image):

    returnDiv = html.Div([
        dbc.Row(
            [
            #html.H5(filename),
            #html.H6(datetime.datetime.fromtimestamp(date)),

            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            dbc.Col
            (
                html.Div([
                    html.H5("Original image"),
                    html.Img(id="original_image",src=contents)
                    ])
            ),
            
            dbc.Col
            (    
                html.Div([  
                    html.H5("Constructed Image"),
                    html.Img(id="constructed_img",src=contructed_image)
                ])

            ),

            ]),


      
    ])
    

    return returnDiv

def analyse_image_fun(contents, filename, date):


    # Decoding string base64 into an image
    content_type, content_string = contents.split(',')
    im = Image.open(BytesIO(base64.b64decode(content_string)))

    # Resize of image and proper datatype
    np_img = np.array(im)
    #size = 512
    size = 256
    np_reshape = np.reshape(im,(1, 3, size, size))
    floatAstype = np.float32(np_reshape)

    # ONNX runtime
    #sess = rt.InferenceSession("dishan_made_unet_model.onnx")
    sess = rt.InferenceSession("dishan_segnet_v2.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[-1].name
    pred_onx = sess.run("",{input_name:floatAstype})

    # size is 256
    
    # Creating the array 
    rgb_array = np.zeros((size,size,3), 'uint8')
  
    # 1*1*3*512*512

    # Choosing class of the highest index 
    # highest probability of each pixel cell 
    highest_index = np.argmax(pred_onx[0][0], axis=0)
    

    for x in range(size):
        for y in range(size):
            index = highest_index[x][y]
            if index == 0:

                # canopy
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 255
                rgb_array[x,y,2] = 0

            elif index == 1:

                # soil
                rgb_array[x,y,0] = 165
                rgb_array[x,y,1] = 42
                rgb_array[x,y,2] = 42

            elif index == 2:

                #stubble 
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 0
                rgb_array[x,y,2] = 255
            else:
 
                #None
                rgb_array[x,y,0] = 255
                rgb_array[x,y,1] = 0
                rgb_array[x,y,2] = 0


    pil_img = Image.fromarray(rgb_array)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    new_image_string = "data:image/JPEG;base64,"+new_image_string
    
    

    return new_image_string

    

# call bacl for upload input 
@app.callback(
    [
        Output(component_id='output-image-upload',component_property='children'),
        Output(component_id="download-content",component_property="href"),
    ],
    [Input(component_id='upload-image', component_property= 'contents')],
    [State('upload-image', 'filename'),
    State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        # children will return a html content
        #children = [
        #    parse_contents(c, n, d) for c, n, d in
        #    zip(list_of_contents, list_of_names, list_of_dates)]  # assigning c,n,d as  inputs and zip version of inputs
        
        
        image_contructed = [
            analyse_image_fun(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
        # Need to pass all the information in creating the divs
        img_html_div_constructed = [
            parse_contents(c, n, d, i) for c, n, d, i in
            zip(list_of_contents, list_of_names, list_of_dates,image_contructed )]  # assigning c,n,d as  inputs and zip version of inputs

        #print(children)
        #print("$$$$$$$$$$$$$$",len(return_values))
        return img_html_div_constructed,image_contructed[0]
    else:
        return '','' # Place holder the when the call back is called in preload





if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server()