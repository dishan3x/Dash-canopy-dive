import dash_bootstrap_components as dbc 
import dash_html_components as html

def app_information():
    return html.Div(
    [
        dbc.Button("About us", id="open-modal-btn-in-about-us-modal"),
        dbc.Modal(
            [
                dbc.ModalHeader("About us"),
                dbc.ModalBody("This app was developed by the programmers"),
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
                dbc.ModalBody(
                    children=[
                        html.H6(children='1 . Select Model'),
                        html.P(),
                        html.H6(children='2 . Upload image'),
                        html.P(),
                        html.H6(children='3 . Download image'),

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