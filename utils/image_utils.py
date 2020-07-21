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
    #'2':"models/dishan_segnet_v2_512.onnx",
    model_dict = {
        '1':"models/dishan_made_simple_segnet_model.onnx",
        '2':"models/dishan_made_unet_model.onnx",
        '3':"models/test.onnx"
    }
    
    # Decoding string base64 into an image
    content_type, content_string = contents.split(',')

    # Decode base64 to bytes
    im = Image.open(BytesIO(base64.b64decode(content_string)))

    # image should be arrange to the model specification
    size        = 512 # size is changed to 256 becuase the model is larger and could not upload to github
     

    #  resize the image to to actual size 
    im_resized  = im.resize((size,size))

    # Convert numpy array for further calculations
    np_img = np.array(im_resized)
    
    try:
        # reshape input
        np_reshape  = np.reshape(np_img,(1, 3, size, size))
        floatAstype = np.float32(np_reshape)

        # Run ONNX runtime
        sess        = rt.InferenceSession(model_dict[selected_model])

        # Retrieving  names of input and output layer of the label
        input_name  = sess.get_inputs()[0].name
        output_name =sess.get_outputs()[0].name

        # Run the input in onnx model
        pred_onx    = sess.run([output_name],{input_name:floatAstype})


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


        # Convert the resized origianl image to base64
        # This process is need to keep the steady size of the input

    

        pil_img_original = Image.fromarray(np_img)
        # using an in-memory bytes buffer
        buff_original    = BytesIO()
        
        pil_img_original.save(buff_original, format="JPEG")
        original_image_string = base64.b64encode(buff_original.getvalue()).decode("utf-8")
        original_image_string = "data:image_ol/JPEG;base64,"+original_image_string

        # convert RGB array to base64
        pil_img = Image.fromarray(rgb_array)
        # use and initiate a different buffer for constructed image 
        buff_constructed    = BytesIO()
        pil_img.save(buff_constructed, format="JPEG")
        contructed_image_str = base64.b64encode(buff_constructed.getvalue()).decode("utf-8")
        contructed_image_str = "data:image/JPEG;base64,"+contructed_image_str


        # Retrieving pixel count in each groups
        # canopy - 0
        # soil   - 1
        # stubble- 2 
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
        img_html_div_constructed = analysed_info_to_html_func(original_image_string, filename, date,contructed_image_str,pixel_count_data)


        return img_html_div_constructed

    except Exception as e:
        print("Misspelled output name")
        print("{0}: {1}".format(type(e), e))
        #return "Sorry ! something went wrong"
        return dbc.Alert(
            children =[
                html.H4("Oops something went wrong!", className="alert-heading"),
                html.P(
                    "{0}: {1}".format(type(e), e)
                ),
                html.Hr(),
            ],
            id="alert-fade",
            color="danger",
            dismissable=True,
            is_open=True,
        )


def analysed_info_to_html_func(contents, filename, date, contructed_image,pixel_count_data):
    '''
        creating the html with analysed data results
    '''

    images_div =dbc.Row(
        id ="image-container-div",
        children =[
            #html.H5(filename),
            #html.H6(datetime.datetime.fromtimestamp(date)),

            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            
            dbc.Card(
                className="image-container",
                children=[
                    dbc.CardImg(id="original-image",src=contents),
                    #html.H5("Original image"),
                    html.A(
                        id="download-content-original",
                        download=str(datetime.datetime.fromtimestamp(date))+"_original.png",
                        href = contents,
                        children=[
                            dbc.Button(
                                "Original image",
                                id="down-load_button",
                                color="primary",
                                className="inline_button",
                                )
                            ]
                        )
                ]
                ),

            dbc.Card(
                className="image-container",
                children=[
                    dbc.CardImg(id="constructed-image",src=contructed_image),
                    #html.H5("Original image"),
                    html.A(
                        id="download-content-construct",
                        download=str(datetime.datetime.fromtimestamp(date))+"_result.png",
                        href = contructed_image,
                        children=[
                            dbc.Button(
                                "Result image",
                                id="down-load_button",
                                color="primary",
                                className="inline_button",
                                )
                            ]
                        )
                ]
                ),                     
        ],
        )
    
    
    percentage_circles =dbc.Row(
            [

            dbc.Col
            (
                html.Div(children=[
                    dbc.Card([
                        html.H6("Canopy",className="percentage-information-cards_title"),
                        dbc.CardBody([
                            html.A(pixel_count_data['canopy_p'],className="result-titles")
                        ]
                        ) 
                    ],
                    className="percentage-information-cards  text-center ",
                    ),
                    ],
                    className="percentage-information-cards-div primary"),
                    sm=4
            ),
            
            dbc.Col
            (
                html.Div(children=[
                    dbc.Card([
                        html.H6("Soil",className="percentage-information-cards_title"),
                        dbc.CardBody([
                            html.A(pixel_count_data['soil_p'],className="result-titles")
                        ]
                        ) 
                    ],
                    className="percentage-information-cards text-center",
                    ),
                    ],
                    className="percentage-information-cards-div"),
                    sm=4
            ),
            

            dbc.Col
            (
                html.Div(children=[
                    dbc.Card([
                        html.H6("Stubble",className="percentage-information-cards_title"),
                        dbc.CardBody([
                            html.A(pixel_count_data['stubble_p'],className="result-titles")
                        ]) 
                    ],
                    className ="percentage-information-cardsS text-center"
                    ),
                    ],
                    className="percentage-information-cards-div"),
                    sm=4
            ),
            

            ]
            )

    # Toast button to show the image process has been complete
    toast = html.Div(
        [
            dbc.Toast(
                [html.P("Image constructed.", className="mb-0")],
                id="complete-toast",
                header="Success",
                duration=2000,
                icon ="success",
                style={"position": "fixed", "top": 66, "right": 10, "width": 250},
            ),
        ]
    )    

    returnDiv = html.Div([
        toast,
        percentage_circles,
        images_div,
    ])
    

    return returnDiv    
