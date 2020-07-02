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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


navbar = dbc.NavbarSimple(
    children=[
        dbc.title('Hey'),
        #dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        #dbc.DropdownMenu(
        #    children=[
        #        dbc.DropdownMenuItem("More pages", header=True),
        #        dbc.DropdownMenuItem("Page 2", href="#"),
        #        dbc.DropdownMenuItem("Page 3", href="#"),
        nav=True,
        in_navbar=True,
        label="More",
        ),
    ],
    brand="NavbarSimple",
    brand_href="#",
    color="primary",
    dark=True,
) 

body = html.Div([
        html.Div(
        id="app-header",
        children=[
            html.Div('Plotly Dash', className="app-header--title")
        ]
    ),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
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


app.layout = html.Div([navbar,body])

def parse_contents(contents, filename, date):


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
    sess = rt.InferenceSession("dishan_made_unet_model.onnx")
    #sess = rt.InferenceSession("dishan_segnet_v2.onnx")
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
    
    # px
    px.imshow(rgb_array)

    return html.Div([
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Div('original image'),
        html.Img(src=contents),

        
        html.H5("Constructed Image"),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        html.Img(src=new_image_string),
  
    ])

    

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children





if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server()