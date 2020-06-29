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


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
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
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
])


def parse_contents(contents, filename, date):


    # Decoding string base64 into an image
    content_type, content_string = contents.split(',')
    im = Image.open(BytesIO(base64.b64decode(content_string)))

    # Resize of image and proper datatype
    np_img = np.array(im)
    size = 512
    np_reshape = np.reshape(im,(1, 3, size, size))
    floatAstype = np.float32(np_reshape)

    # ONNX runtime
    sess = rt.InferenceSession("dishan_made_unet_model.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[-1].name
    pred_onx = sess.run("",{input_name:floatAstype})

    # size is 256
x


    # Creating the array 
    rgb_array = np.zeros((size,size,3), 'uint8')
    #red = rgb_array[:,:,0]
    #green = rgb_array[:,:,1]
    #blue = rgb_array[:,:,2]
    # 1*1*3*512*512
    for x in range(width):
        for y in range(height):
            index = 0
            max_prediction = 0
            for i in range(0,3):
                pred = pred_onx[0][0][i][x][y]
                if pred > max_prediction:
                    index = i
                    max_prediction = pred
            

            if index == 0:

                label_holder = "canopy"
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 255
                rgb_array[x,y,2] = 0

            elif index == 1:

                label_holder = "soil"
                rgb_array[x,y,0] = 165
                rgb_array[x,y,1] = 42
                rgb_array[x,y,2] = 42

            elif index == 2:

                label_holder = "stubble" 
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 0
                rgb_array[x,y,2] = 255
            else:
 
                label_holder = "None"
                rgb_array[x,y,0] = 255
                rgb_array[x,y,1] = 0
                rgb_array[x,y,2] = 0




    pil_img = Image.fromarray(rgb_array)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    new_image_string = "data:image/JPEG;base64,"+new_image_string
    
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
        
        html.H5("Constructed Image"),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        html.Img(src=new_image_string),
        html.Hr(),
        html.Div('Raw Content'),
        html.Pre(new_image_string[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
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