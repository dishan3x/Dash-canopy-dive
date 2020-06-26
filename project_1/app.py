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
    print()
    print("hello")
    print(contents)
    print()
    content_type, content_string = contents.split(',')

    #sess = rt.InferenceSession("dishan_segnet_v2.onnx")
    sess = rt.InferenceSession("dishan_made_unet_model.onnx")
    print("****************************************")
    # encode frame 
    #encoded_string = base64.b64encode(contents.read())
    # decode frame
    #decoded_string = base64.b64decode(encoded_string)
    #decoded_img = np.fromstring(decoded_string, dtype=np.uint8)
    #decoded_img = decoded_img.reshape(contents.shape)
    decoded = base64.b64decode(content_string)
    #print(decoded)

    #im1 = Image.open(decoded)
    #print(im1)
    im = Image.open(BytesIO(base64.b64decode(content_string)))

    print(im)


    
    pix = im.load()
    print("******pix**********************************")
    print(pix)

    np_img = np.array(im)
    size = 512
    print("np_img.shape",np_img.shape)
    np_reshape = np.reshape(im,(1, 3, size,size))
    floatAstype = np.float32(np_reshape)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[-1].name

    # Run the model
    pred_onx = sess.run("",{input_name:floatAstype})

    #print(pred_onx)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$")
    #print(pred_onx[0][0][0][0][0])
    #print(pred_onx[0][0][1][0][0])
    #print(pred_onx[0][0][2][0][0])


    # size is 256
    width = size
    height = size


    # Creating the array 

    image_construced = np.empty(shape=[width, height])
    max_probability_holder = np.empty(shape=[width, height])
    label_holder = np.empty(shape=[width, height])

    for x in range(width):
        for y in range(height):
            index = 0
            max_prediction = 0
            for i in range(0,3):
                pred = pred_onx[0][0][i][x][y]
                if pred > max_prediction:
                    index = i
                    max_prediction = pred
            
            
            max_probability_holder[x][y] = max_prediction
            
            # ignore the rest of the max_prediction prediction portion
            
            # End of for loop for each row of 4 softmax probabilities
            # Then we check with weights to see which catogory its in 
            
            # Blue_per   =   0.3573605344575649
            # Pink_per   =   0.3228344583030248
            # Yellow_per =   0.31691428485563417
            # Bed_per    =  0.0028907223837762435
        
            if max_prediction > 0.3573605344575649:
                image_construced[x][y] = 16448250  # 1
                label_holder = "canopy"
            elif max_prediction >  0.3228344583030248:
                image_construced[x][y] = 50000 # 2
                label_holder = "soil"
            elif max_prediction >  0.31691428485563417:
                image_construced[x][y] = 6000   # 3
                label_holder = "stubble" 
            else:
                image_construced[x][y] = 0   # 4
                label_holder = "None"

    #print(max_probability_holder[0][0])
    #print(max_probability_holder[0][1])
    #print(max_probability_holder[0][2])
    
    #imgloo = Image.fromarray(image_construced)                  #Crée une image à partir de la matrice
    #buffer = BytesIO()
    #imgloo.save(buffer,format="JPEG")                  #Enregistre l'image dans le buffer
    #myimage = buffer.getvalue() 
    #image_construced = image_construced.astype(np.uint8)
    pil_img = Image.fromarray(image_construced)
    if pil_img.mode != 'RGB':
       pil_img = pil_img.convert('RGB')
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    new_image_string = "data:image/JPEG;base64,"+new_image_string
    #data:image/png;base64,
    print("BUfffffffffffffffffffffffffffffffffffffffffffererer")
    #print(new_image_string)
    print("555Endeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    #print(myimage)
    #sd = ""
    #sd = base64.b64encode(myimage)
    #srtt = "data:image/jpeg;base64,"+sd.decode('utf-8')
    #print(srtt)
    print("here") ;    
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