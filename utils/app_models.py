import dash_bootstrap_components as dbc 
import dash_html_components as html

def app_information():
    return html.Div(
    [
        dbc.Button("About us", id="open-modal-btn-in-about-us-modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("About us"),
                dbc.ModalBody([
                    html.P(
                    ["Cora is an applicaiton program that can quantify the soil cover by processing/analyzing an image input from the user. The appliation can quantify the soil cover on the basis of soil,residue and canopy. ",
                    html.Br(),html.Br(),
                    "Soil residues are stems and stalks that remain on soil from previous crops. Residue cover is utilized in farming techniques such as no-till farming. This farming technique uses residue as a cover to the soil layer. It acts as a barrier to the soil by deflecting energy from the raindrops that can wash away the soil particles and the nutrients in the soil. It also reduce the soil erosion and preserve water level drying out from sun.",
                    html.Br(),html.Br(),
                    "Identify the residue cover in the soil has higher importance. There percentages of soil cover that maintain throughout farming process. Moreover it also useful research purposes.",
                    html.Br(),html.Br(),
                    "Technology improvement world has have major impact in many fields. We uses a latest computer models to analyse the image data. Our effort is to minimize and save the time and energy in calculating soil cover. "
                    "Hope this app will make your life easier.",
                    html.Br(),html.Br(),
                    #"References",
                    html.Br(),html.Br(), 
                    ]),
                ]   
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="close-modal-btn-in-about-us-modal", className="ml-auto")
                ),
            ],
            id="modal_about_us",
        ),
    ]
    )

def app_instructions():
    return html.Div(
    [
        dbc.Button("Instructions", id="open-modal-btn-in-instructions-modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("Instructions"),
                dbc.ModalBody([
                        html.H5('1 . Take an image',className='instruction-list-header'),
                        html.Div([
                        html.Li("Stand on soil where you need to calculate soil cover."),
                        html.Li("Open the cora app and click on upload image button to open the camera"),
                        html.Li("Hold the mobile camera 1m above ground level."),
                        html.Li("Capture image"),
                        ],className="instruction-sublist"),
                        html.Br(),
                        html.H5(children='2 . Analysed image',className='instruction_list_header'),
                        html.Div([
                        html.Li("Calculated the soil cover percentages will show on top of the application."),
                        html.Li("Application will be using color schemes to visualized the analysed data."),
                        ],className="instruction-sublist"),
                        html.Br(),
                        html.H5(children='3 . Download image',className='instruction_list_header'),
                        html.Div([
                        html.Li("Click on the download button to download the images."),
                        ],className="instruction-sublist"),
                                                html.Br(),
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
