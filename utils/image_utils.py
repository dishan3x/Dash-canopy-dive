import time

start = time.time()
import datetime
import dash_html_components as html
import numpy as np
import PIL
from PIL import Image,ImageOps
import base64
from io import BytesIO
import dash_bootstrap_components as dbc
import tensorflow as tf
from tensorflow.keras.models import model_from_json


print("------- %s loading time for libraries in seconds ---" % (time.time() - start))


def analyse_image_func(contents, filename, date,selected_model):

    """
        This fuction run the image through the onnx model 
        and return the prediction relating to the classes
        provided from the model. 
    """
    #'2':"models/dishan_segnet_v2_512.onnx",
    #model_dict = {
    #    '1':"models/dishan_made_simple_segnet_model.onnx",
    #    '2':"models/dishan_made_unet_model.onnx",
    #    '3':"models/test.onnx"
    #}

    #model_dict = {
    #    '1':"models/Sample_model.onnx",
    #    '2':"models/Sample_model.onnx",
    #    '3':"models/Sample_model.onnx"
    #}
    begin = time.time()
    start = time.time()
    
    # Decoding string base64 into an image
    content_type, content_string = contents.split(',')

    # Decode base64 to bytes
    im = Image.open(BytesIO(base64.b64decode(content_string)))
    #print("New request --- %s image stored in varaible completed seconds ---" % (time.time() - start))
    #print("xxxxxx",im.getdata())
    #colours, col_counts = np.unique(im.reshape(-1,3), axis=0, return_counts=1)
    #print("colors",colours)
    #print(col_counts)
    print(type(im))
    # image should be arrange to the model specification
    size        = 512 # size is changed to 256 becuase the model is larger and could not upload to github
     
    start = time.time()
    #print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPP size",im.shape)
    #  resize the image to to actual size 
    #im_resized  = im.resize((size,size))
    #im_resized = im.crop((512, 512, 512, 512)) 
    #print("**********************",im_resized.shape)

    # Convert numpy array for further calculations
    basewidth = 512

    np_evaluate = np.array(im)
    print("After**********",np_evaluate.shape)
    im_resized = None
    if(np_evaluate.shape[0] < size):
        wpercent = (basewidth / float(np_evaluate.shape[0]))
        hsize = int((float(np_evaluate.shape[1]) * float(wpercent)))
        #im_resized = im.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        im_resized    = im.resize((size,size))
    
    else:
        im_resized    = im.resize((size,size))

    print("refitting")
    #size = (512, 512)
    #fit_and_resized_image = ImageOps.fit(np_evaluate, size, Image.ANTIALIAS)
    #print("shape refittig",np.array(fit_and_resized_image).shape)
    
    #im1 = img.crop((512, 512, 512, 512)) 
    #np_kpkp = np.array(im1)
    #print("after**********",np_kpkp.shape)    

    #print("before****2******",np_img.shape)
    

    np_img = np.array(im_resized)
    
    
    print("After**********",np_img.shape)
    #print("After****2******",np.array(im_resized).shape)



    #im_thumbnail  = im.thumbnail((size,size), PIL.Image.ANTIALIAS)
    #print("--- %s image resized and used as numpy  seconds ---" % (time.time() - start))
    #im_thumbnailP = np.array(im_thumbnail)
    #print("after**thumb********",im_thumbnailP.shape)   

    #kol = np.transpose(im,(0, 1, 2))
    #iol = np.expand_dims(np_img, 0)

    #kol = np.transpose(im,(3, 0,1,2))
    #np_img = np.array(image)
    #kol = np.transpose(im,(0, 1,2))
    #floatAstype = np.float32(np_img)

    #reshape_image  = np.expand_dims(im, 0)
    #im = Image.open("img_1.png")

    # reshape input
    #np_ar = np.array(im_resized)
    #np_reshaped  = np.reshape(np_ar,(size, size,3))
    #np_reshape  = np.reshape(iol,(1,size, size,3))
    #np_reshape = np.transpose(im,(2,1,0))
    #print("np_reshape",np_reshape.shape)
    #np.expand_dims(im, 0)

    #https://github.com/plotly/dash-image-processing/blob/master/notebooks/Exploring%20PIL%20Processing.ipynb
    floatAstype = np.float32(np.expand_dims(np_img,0))
    #selected_model = 'models/Sample_model.onnx'


    try:
        # reshape input
        #print("come here",floatAstype.shape)
        #np_reshape  = np.reshape(np_img,(1,size, size,3))
        #floatAstype = np.float32(iol)
        
        #print("come here 2",floatAstype.shape)

        # Run ONNX runtime
        #sess        = rt.InferenceSession(selected_model)

        # Retrieving  names of input and output layer of the label
        #input_name  = sess.get_inputs()[0].name
        #output_name =sess.get_outputs()[0].name
        #print("shape",floatAstype.shape)

        # Run the input in onnx model
        #pred_onx    = sess.run([output_name],{input_name:floatAstype})


        # Creating the image array 
        rgb_array = np.zeros((size,size,3), 'uint8')
    
        # Choosing class of the highest index 
        # highest probability of each pixel cell 
        start = time.time()
        #new_model = tf.keras.models.load_model('models/save_model_file')
        
        json_file = open("models/json_model/model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        new_model = model_from_json(loaded_model_json)
        new_model.load_weights("models/json_model/model_3000.h5")
        
        print("--- %s model loaded using tf seconds ---" % (time.time() - start))
        start = time.time()
        pred_onx = new_model.predict(floatAstype)
        print("--- %s prediction calculated via image seconds ---" % (time.time() - start))
        start = time.time()
        highest_probability_index = np.argmax(pred_onx[0], axis=2)
        print("--- %s probability calulated using argmax seconds ---" % (time.time() - start))
        # convert prediction array to RGB image.
        
        start = time.time()

        #update the rows and cols ---------------------------------------------------->
        # Get all the index  stubble 
        idx_stubble = highest_probability_index == 0 
        rgb_array[idx_stubble,0]= 0
        rgb_array[idx_stubble,1]= 0 
        rgb_array[idx_stubble,2]= 255  

        idx_stubble = highest_probability_index == 1
        rgb_array[idx_stubble,0]= 165
        rgb_array[idx_stubble,1]= 42 
        rgb_array[idx_stubble,2]= 42 

        idx_stubble = highest_probability_index == 2
        rgb_array[idx_stubble,0]= 0
        rgb_array[idx_stubble,1]= 255 
        rgb_array[idx_stubble,2]= 0 


        #for x in range(size):
        #    for y in range(size):

        #        index = highest_probability_index[x][y]
                
        #        if index == 2:

                    # canopy -> green
        #            rgb_array[x,y,0] = 0
        #            rgb_array[x,y,1] = 255
        #            rgb_array[x,y,2] = 0

        #        elif index == 1:

                    # soil -> brown
        #            rgb_array[x,y,0] = 165
        #            rgb_array[x,y,1] = 42
        #            rgb_array[x,y,2] = 42

        #        elif index == 0:

                    #stubble  -> blue
        #            rgb_array[x,y,0] = 0
        #            rgb_array[x,y,1] = 0
        #            rgb_array[x,y,2] = 255


        # Convert the resized origianl image to base64
        # This process is need to keep the steady size of the input

        print("--- %s RGB image created in seconds ---" % (time.time() - start))
        start = time.time()
        pil_img_original = Image.fromarray(np_img)
        #pil_img_original = pil_img_original.resize(size,size)
        #pil_img_original = im_resized
        # using an in-memory bytes buffer
        buff_original    = BytesIO()
        print("arrive here 1")
        pil_img_original.save(buff_original, format="PNG")
        original_image_string = base64.b64encode(buff_original.getvalue()).decode("utf-8")
        #original_image_string = base64.b64encode(im_resized.tobytes())
        original_image_string = "data:image_ol/JPEG;base64,"+original_image_string
        print("arrive here 2")
        # convert RGB array to base64
        # Transpose RGB array 
        
        pil_img = Image.fromarray(rgb_array)
        #rgb_transpose = pil_img.transpose(Image.ROTATE_45)
        #Transpose
        #rgb_transpose = pil_img.transpose(Image.TRANSPOSE)

        #(background, overlay, 0.5)
        blend = Image.blend(pil_img_original, pil_img, 0.5)
        # use and initiate a different buffer for constructed image 
        buff_constructed    = BytesIO()
        blend.save(buff_constructed, format="JPEG")
        print("arrive here 3")
        contructed_image_str = base64.b64encode(buff_constructed.getvalue()).decode("utf-8")
        contructed_image_str = "data:image/JPEG;base64,"+contructed_image_str

        print("--- %s images constructed ---" % (time.time() - start))


        # Retrieving pixel count in each groups
        # canopy - 0
        # soil   - 1
        # stubble- 2 
        print("sds",highest_probability_index.shape)
        unique_group_name, counts = np.unique(highest_probability_index, return_counts=True)
        print("sds",unique_group_name)
        pixel_spread = dict(zip(unique_group_name, counts))
        
        # Retreive the sum of pixel 
        sum_pixel           = sum(counts) 
        # percentages for each group
        stubble_percentage = 0
        canopy_percentage = 0
        soil_percentage = 0

        if 0 in pixel_spread:
            stubble_percentage =  "{:.1%}".format(pixel_spread[0]/sum_pixel)
        if 1 in pixel_spread:
            soil_percentage = "{:.1%}".format(pixel_spread[1]/sum_pixel)
        if 2 in pixel_spread:
            canopy_percentage = "{:.1%}".format(pixel_spread[2]/sum_pixel)


        #stubble_percentage  = "{:.1%}".format(pixel_spread[0]/sum_pixel)
        #canopy_percentage= "{:.1%}".format(pixel_spread[2]/sum_pixel)
        #soil_percentage= "{:.1%}".format(pixel_spread[1]/sum_pixel)
        print("sds",highest_probability_index.shape)
        print(stubble_percentage)
        #print(canopy_percentage)
        #print(soil_percentage)

        pixel_count_data = {
            'canopy_p':canopy_percentage,
            'soil_p':stubble_percentage,
            'stubble_p':soil_percentage
            }

        start = time.time()

        # Create an ar
        img_html_div_constructed = analysed_info_to_html_func(original_image_string, filename, date,contructed_image_str,pixel_count_data)

        print("--- %s HTML content construct time in S ---" % (time.time() - start))
        print(">>> %s probability overall seconds ---" % (time.time() - begin))

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
                        dbc.CardBody([
                            html.P("Canopy cover %",id="canopy-title",className="percentage-information-cards_title"),
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
                        dbc.CardBody([
                            html.P("Soil cover %",id="residue-title",className="percentage-information-cards_title"),
                            html.A(pixel_count_data['stubble_p'],className="result-titles")
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
                        dbc.CardBody([
                            html.P("Residue Cover %",id="soil-title",className="percentage-information-cards_title"),
                            html.A(pixel_count_data['soil_p'],className="result-titles")
                        ]) 
                    ],
                    className ="percentage-information-cards text-center"
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
