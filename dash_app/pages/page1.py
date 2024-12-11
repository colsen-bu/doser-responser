import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback, no_update
import pandas as pd
import base64
import io
import dash

layout = dbc.Container([
    html.H1('Upload CSV File', style={'textAlign': 'center'}),

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select CSV File')
        ]),
        style={
            'width': '60%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '0 auto 20px',  # Center and add bottom margin
        },
        multiple=False
    ),

    dbc.Button(
        'Calculate Dose Response',
        id='calculate-button',
        n_clicks=0,
        color='primary',
        style={
            'width': '60%',
            'margin': '0 auto 20px',  # Center the button and add margin below
            'display': 'block'
        }
    ),

    #dcc.Location(id='url', refresh=False),  # Add this line

    html.Div(id='output-data-upload'),  # Div to display the plate visualization

], fluid=True)

# First Callback: Handle file upload and store data
@callback(
    Output('shared-data', 'data'),
    Input('upload-data', 'contents'),
    prevent_initial_call=True
)
def store_uploaded_data(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except Exception as e:
            return no_update  # Handle the error as needed

        # Store the data in shared-data
        data = df.to_dict('records')
        return data
    else:
        return no_update

# Second Callback: Generate plate visualization when shared-data or URL is updated
@callback(
    Output('output-data-upload', 'children'),
    [Input('shared-data', 'data'),
     Input('url', 'pathname')]  # Include URL pathname as Input
)
def update_plate_visualization(data, pathname):
    if pathname == '/page1' and data is not None:
        df = pd.DataFrame(data)
        # Generate plate visualizations as before
        experiment_ids = df['Experiment_ID'].unique()
        plates_visualizations = []
        for experiment_id in experiment_ids:
            experiment_df = df[df['Experiment_ID'] == experiment_id]
            plate_visual = generate_plate_visualization(experiment_df)
            plates_visualizations.append(html.Div([
                html.H3(f'Experiment ID: {experiment_id}'),
                plate_visual
            ], style={'margin-bottom': '40px'}))
        return plates_visualizations
    else:
        return ''  # Return empty if no data or not on Page 1

# Callback to navigate to page2 when button is clicked
@callback(
    Output('url', 'pathname'),
    Input('calculate-button', 'n_clicks'),
    State('shared-data', 'data'),
    prevent_initial_call=True
)
def navigate_to_page2(n_clicks, data):
    if n_clicks > 0:
        return '/page2'
    else:
        return no_update

def generate_plate_visualization(experiment_df):
    rows = ['A','B','C','D','E','F','G','H']
    cols = [str(i) for i in range(1,13)]
    
    # Create header row with column numbers
    header_row = [html.Th('', style={'text-align': 'center'})] + \
                 [html.Th(col, style={'text-align': 'center'}) for col in cols]
    plate_layout = [html.Tr(header_row)]

    for row in rows:
        row_cells = [html.Th(row, style={'text-align': 'center'})]  # Center row letters
        for col in cols:
            well = f'{row}{col}'
            cell_data = experiment_df[experiment_df['Well'] == well]
            if not cell_data.empty:
                cell_content = html.Div([
                    html.P(f'{cell_data.iloc[0]["Treatment"]}'),
                    html.P(f'{cell_data.iloc[0]["Dose_uM"]} μM'),
                ], style={'font-size': '10px', 'text-align': 'center'})
                cell_style = {
                    'border': '1px solid black',
                    'width': '60px',
                    'height': '60px',
                    'background-color': '#0e2f44',
                }
            else:
                cell_content = html.Div()
                cell_style = {
                    'border': '1px solid black',
                    'width': '60px',
                    'height': '60px',
                    'background-color': '#494949',
                }
            row_cells.append(html.Td(cell_content, style=cell_style))
        plate_layout.append(html.Tr(row_cells))
    
    table = html.Table(plate_layout, style={
        'border-collapse': 'collapse',
        'margin': 'auto',
    })
    return table