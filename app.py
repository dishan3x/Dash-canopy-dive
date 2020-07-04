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



#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css','dbc.themes.BOOTSTRAP']
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


navbar = dbc.NavbarSimple(
    brand="Cora",
    color="primary",
    dark=True,
) 

body = dbc.Container([
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
app.layout = html.Div([navbar,body])

def analyse_image_func(contents, filename, date):


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
    highest_probability_index = np.argmax(pred_onx[0][0], axis=0)
    

    for x in range(size):
        for y in range(size):
            index = highest_probability_index[x][y]
            if index == 0:

                # canopy -> green
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 255
                rgb_array[x,y,2] = 0

            elif index == 1:

                # soil -> brown
                rgb_array[x,y,0] = 165
                rgb_array[x,y,1] = 42
                rgb_array[x,y,2] = 42

            elif index == 2:

                #stubble  -> blue
                rgb_array[x,y,0] = 0
                rgb_array[x,y,1] = 0
                rgb_array[x,y,2] = 255



    # convert RGB array to base64
    pil_img = Image.fromarray(rgb_array)
    buff    = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    new_image_string = "data:image/JPEG;base64,"+new_image_string


    # Retrieving pixel count in each groups
    # canopy - 0
    # soil   - 1
    # stubble- 3 
    unique_group_name, counts = np.unique(highest_probability_index, return_counts=True)
    pixel_spread = dict(zip(unique_group_name, counts))

    # Retreive the sum of pixel - 
    # ex : 512 * 512 =  
    sum_pixel           = sum(counts) 

    # percentages for each group
    canopy_percentage   = round((pixel_spread[0]/sum_pixel),2)
    soil_percentage     = round((pixel_spread[1]/sum_pixel),2)
    stubble_percentage  = round((pixel_spread[2]/sum_pixel),2)


    pixel_count_data = {
        'canopy_p':canopy_percentage,
        'soil_p':soil_percentage,
        'stubble_p':stubble_percentage
        }

    # Create an ar
    img_html_div_constructed = analysed_info_tohtml_func(contents, filename, date,new_image_string,pixel_count_data)


    return img_html_div_constructed



def analysed_info_tohtml_func(contents, filename, date, contructed_image,pixel_count_data):
    '''
        creating the html with the analysed data
    '''



    images_div =dbc.Row(
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

            ]
            )

       
    # creating table for the percentage values
    table_header = [
        html.Thead(html.Tr([html.Th("Type"), html.Th("Percentage")]))
    ]

    row1 = html.Tr([html.Td("Canopy"), html.Td(pixel_count_data['canopy_p'])])
    row2 = html.Tr([html.Td("Soil"), html.Td(pixel_count_data['soil_p'])])
    row3 = html.Tr([html.Td("Stubble"), html.Td(pixel_count_data['stubble_p'])])
 

    table_body = [html.Tbody([row1, row2, row3])]

  
    table = dbc.Table(
        # using the same table as in the above example
        table_header + table_body,
        id ="percentage_table",
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    download_button = html.A(
            id="download-content",
            download="image.png",
            href = contructed_image,
            children=[
            dbc.Button(
                "Download image",
                id="down-load_button",
                color="primary",
                className="inline_button",
            )
            ])  

    returnDiv = html.Div([
            images_div,
            table,
            download_button,
    ])
    

    return returnDiv    
    
# call bacl for upload input 
@app.callback(Output(component_id='output-image-upload',component_property='children'),
    [Input(component_id='upload-image', component_property= 'contents')],
    [State('upload-image', 'filename'),
    State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):

    if list_of_contents is not None:
        analysed_information = [
            analyse_image_func(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return analysed_information
    else:
        return '' # Place holder the when the call back is called in preload





if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server()