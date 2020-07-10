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


def analyse_image_func(contents, filename, date,selected_model):

    """
        This fuction run the image through the onnx model 
        and return the prediction relating to the classes
        provided from the model. 
    """
    model_dict = {
        '1':"models/dishan_made_simple_segnet_model.onnx",
        '2':"models/dishan_segnet_v2.onnx",
        '3':"soon.onnx"
    }
    
    # Decoding string base64 into an image
    content_type, content_string = contents.split(',')

    # Decode base64 to bytes
    im = Image.open(BytesIO(base64.b64decode(content_string)))

    # image should be arrange to the model specification
    size        = 512 # size is changed to 256 becuase the model is larger and could not upload to github
    
    # resize the image to to actual size 
    im_resized  = im.resize((size,size))

    # Convert numpy array for further calculations
    np_img = np.array(im_resized)
    
    try:
        # reshape input
        np_reshape  = np.reshape(np_img,(1, 3, size, size))
        floatAstype = np.float32(np_reshape)

        # Run ONNX runtime
        sess        = rt.InferenceSession(model_dict[selected_model])
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[-1].name

        # Run the input in onnx model
        pred_onx    = sess.run("",{input_name:floatAstype})


        # Creating the image array 
        rgb_array = np.zeros((size,size,3), 'uint8')
    
        
        # Choosing class of the highest index 
        # highest probability of each pixel cell 
        highest_probability_index = np.argmax(pred_onx[0][0], axis=0)
        
        # convert prediction array to RGB image.
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

        # Retreive the sum of pixel 
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
        img_html_div_constructed = analysed_info_to_html_func(contents, filename, date,new_image_string,pixel_count_data)


        return img_html_div_constructed

    except Exception as e:
        print("Misspelled output name")
        print("{0}: {1}".format(type(e), e))
        return "Sorry ! something went wrong"



def analysed_info_to_html_func(contents, filename, date, contructed_image,pixel_count_data):
    '''
        creating the html with analysed data results
    '''

    images_div =dbc.Row(
            [
            #html.H5(filename),
            #html.H6(datetime.datetime.fromtimestamp(date)),

            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            
            html.Div(
                className="image-container",
                children=[
                    html.H5("Original image"),
                    html.Img(id="original-image",src=contents),
                    html.A(
                        id="download-content-original",
                        download="image.png",
                        href = contents,
                        children=[
                            dbc.Button(
                                "Download image",
                                id="down-load_button",
                                color="primary",
                                className="inline_button",
                                )
                            ]
                        )  
                ]
                ),

            html.Div(
                className="image-container",
                children=[
                    html.H5("Constructed Image"),
                    html.Img(id="constructed-image",src=contructed_image),
                    html.A(
                        id="download-content-construct",
                        download="image.png",
                        href = contructed_image,
                        children=[
                            dbc.Button(
                                "Download image",
                                id="down-load_button",
                                color="primary",
                                className="inline_button",
                                )
                        ]
                        ) 
                ]
                ),
            
            ]
        )
    
    
    percentage_circles =dbc.Row(
            [

            dbc.Col
            (
                html.Div(children=[
                    html.H6("Canopy"),
                    html.A(pixel_count_data['canopy_p'],className="result-titles")
                    ],
                    className="rounded-circle")
            ),
            
            dbc.Col
            (    
                html.Div(children=[  
                    html.H6("Soil"),
                    html.A(pixel_count_data['soil_p'],className="result-titles")
                ],
                className="rounded-circle")

            ),

            dbc.Col
            (
                html.Div(children=[
                    html.H6("Stubble"),
                    html.A(pixel_count_data['stubble_p'],className="result-titles")
                    ],
                    className="rounded-circle")
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
        #dark=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    #download_button = html.A(
    #        id="download-content",
    #        download="image.png",
    #        href = contructed_image,
    #        children=[
    #        dbc.Button(
    #            "Download image",
    #            id="down-load_button",
    #            color="primary",
    #            className="inline_button",
    #        )
    #        ])  

    returnDiv = html.Div([
            percentage_circles,
            images_div,
            #download_button,
    ])
    

    return returnDiv    
