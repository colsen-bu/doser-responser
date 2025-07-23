# index.py

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, MATCH, ALL
import dash_bootstrap_components as dbc
from dash_app.app import app, server
import pandas as pd
import numpy as np
import base64
import io
import os
import plotly.graph_objs as go
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
import json
from scipy.optimize import curve_fit
import re

# Load the 'slate' theme
load_figure_template('slate')

# Define plate visualization functions for both icon and full views
def detect_plate_type(experiment_df):
    """Detect whether this is a 96-well or 384-well plate based on well names"""
    wells = experiment_df['Well'].unique()
    
    # Check for 384-well plate patterns (A1-P24)
    has_384_wells = any(well[0] in 'IJKLMNOP' for well in wells if len(well) >= 2)
    max_col = max([int(re.findall(r'\d+', well)[0]) for well in wells if re.findall(r'\d+', well)], default=0)
    
    if has_384_wells or max_col > 12:
        return '384'
    else:
        return '96'

def get_plate_dimensions(plate_type):
    """Get row and column definitions for different plate types"""
    if plate_type == '384':
        rows = [chr(ord('A') + i) for i in range(16)]  # A-P
        cols = [str(i) for i in range(1, 25)]  # 1-24
        return rows, cols
    else:  # 96-well
        rows = ['A','B','C','D','E','F','G','H']
        cols = [str(i) for i in range(1, 13)]  # 1-12
        return rows, cols

def generate_plate_icon(experiment_df, experiment_id, excluded_wells=None):
    """Generate a small icon representation of the plate"""
    if excluded_wells is None:
        excluded_wells = []
    
    plate_type = detect_plate_type(experiment_df)
    total_wells = len(experiment_df)
    excluded_count = len([w for w in excluded_wells if experiment_df['Well'].str.contains(w.split('_')[0] if '_' in w else w).any()])
    
    # Create a small visual representation
    icon_style = {
        'width': '120px',
        'height': '80px',
        'border': '2px solid #0e2f44',
        'border-radius': '8px',
        'background': 'linear-gradient(45deg, #0e2f44 25%, #1a4a5e 25%, #1a4a5e 50%, #0e2f44 50%)',
        'background-size': '10px 10px',
        'display': 'flex',
        'flex-direction': 'column',
        'justify-content': 'center',
        'align-items': 'center',
        'cursor': 'pointer',
        'margin': '10px',
        'transition': 'all 0.3s ease',
        'position': 'relative'
    }
    
    # Add expand indicator using unicode character instead of FontAwesome
    icon_content = html.Div([
        html.Div(f"{plate_type}-well", style={
            'font-weight': 'bold', 
            'color': 'white', 
            'font-size': '12px',
            'margin-bottom': '2px'
        }),
        html.Div(f"{total_wells} wells", style={
            'color': '#aaa', 
            'font-size': '10px',
            'margin-bottom': '2px'
        }),
        html.Div(f"{excluded_count} excluded", style={
            'color': '#ff6b6b' if excluded_count > 0 else '#4ecdc4', 
            'font-size': '9px'
        }),
        # Add expand indicator using unicode
        html.Div("⤢", style={
            'position': 'absolute',
            'top': '5px',
            'right': '8px',
            'color': '#aaa',
            'font-size': '12px'
        })
    ], 
    id={'type': 'plate-icon', 'experiment': experiment_id},
    style=icon_style
    )
    
    return icon_content

def generate_full_plate_visualization(experiment_df, excluded_wells=None, plate_type=None):
    """Generate the full detailed plate visualization"""
    if excluded_wells is None:
        excluded_wells = []
    
    if plate_type is None:
        plate_type = detect_plate_type(experiment_df)
    
    rows, cols = get_plate_dimensions(plate_type)
    
    # Adjust cell sizes based on plate type
    if plate_type == '384':
        cell_width = '35px'
        cell_height = '30px'
        font_size = '6px'
        padding = '1px'
    else:
        cell_width = '70px'
        cell_height = '65px'
        font_size = '9px'
        padding = '2px'
    
    # Create header row with column numbers
    header_row = [html.Th('', style={'text-align': 'center', 'font-size': font_size})]
    
    # For 384-well plates, show fewer column labels to avoid clutter
    if plate_type == '384':
        # Show every 3rd column number
        for i, col in enumerate(cols):
            if i % 3 == 0 or col in ['1', '24']:
                header_row.append(html.Th(col, style={'text-align': 'center', 'font-size': font_size}))
            else:
                header_row.append(html.Th('', style={'text-align': 'center', 'font-size': font_size}))
    else:
        header_row.extend([html.Th(col, style={'text-align': 'center', 'font-size': font_size}) for col in cols])
    
    plate_layout = [html.Tr(header_row)]

    for row in rows:
        row_cells = [html.Th(row, style={'text-align': 'center', 'font-size': font_size})]
        for col in cols:
            well = f'{row}{col}'
            cell_data = experiment_df[experiment_df['Well'] == well]
            
            if not cell_data.empty:
                # Check if this well is excluded
                is_excluded = well in excluded_wells
                
                # Create cell content with text decoration if excluded
                text_style = {'text-decoration': 'line-through'} if is_excluded else {}
                
                if plate_type == '384':
                    # Simplified content for 384-well plates
                    cell_content = html.Div([
                        html.Div(f'{cell_data.iloc[0]["Treatment"][:3]}', 
                                style={'font-weight': 'bold', 'margin-bottom': '1px', **text_style}),
                        html.Div(f'{cell_data.iloc[0]["Response_Metric"]:.1f}', 
                                style={'font-size': '5px', **text_style}),
                    ], style={'font-size': font_size, 'text-align': 'center', 'line-height': '1.0'})
                else:
                    # Full content for 96-well plates
                    cell_content = html.Div([
                        html.Div(f'{cell_data.iloc[0]["Treatment"]}', 
                                style={'font-weight': 'bold', 'margin-bottom': '1px', **text_style}),
                        html.Div(f'{cell_data.iloc[0]["Dose_uM"]}μM', 
                                style={'font-size': '8px', 'margin-bottom': '1px', **text_style}),
                        html.Div(f'{cell_data.iloc[0]["Response_Metric"]:.2f}', 
                                style={'font-size': '8px', **text_style}),
                    ], style={'font-size': font_size, 'text-align': 'center', 'line-height': '1.0'})
                
                cell_style = {
                    'border': '1px solid black',
                    'width': cell_width,
                    'height': cell_height,
                    'background-color': '#0a1d2a' if is_excluded else '#0e2f44',
                    'cursor': 'pointer',
                    'padding': padding
                }
                
                # Add a unique ID for this cell to track clicks
                cell_id = {'type': 'well-cell', 'well': well, 'experiment': cell_data.iloc[0]['Experiment_ID']}
            else:
                cell_content = html.Div()
                cell_style = {
                    'border': '1px solid black',
                    'width': cell_width,
                    'height': cell_height,
                    'background-color': '#494949',
                }
                cell_id = None
                
            # Make each cell a clickable div if it has data
            if cell_id:
                row_cells.append(html.Td(
                    html.Div(cell_content, id=cell_id, style={'width': '100%', 'height': '100%'}), 
                    style=cell_style
                ))
            else:
                row_cells.append(html.Td(cell_content, style=cell_style))
                
        plate_layout.append(html.Tr(row_cells))
    
    table = html.Table(plate_layout, style={
        'border-collapse': 'collapse',
        'margin': 'auto',
    })
    return table


# Create the merged single-page layout
app.layout = dbc.Container([
    
    # Stores for data persistence
    dcc.Store(id='shared-data', storage_type='memory'),
    dcc.Store(id='calculation-state', storage_type='memory', data={'show_visualizations': False}),
    dcc.Store(id='excluded-wells', storage_type='memory', data=[]),  # Store for excluded wells
    dcc.Store(id='curve-fit-data', storage_type='memory', data={}),  # Store for curve fitting results
    dcc.Store(id='active-tab', storage_type='memory', data="tab-combined"),
    dcc.Store(id='expanded-plates', storage_type='memory', data=[]),  # Store for tracking which plates are expanded
    
    # Main content - two column layout
    dbc.Row([
        # Left column - Upload and Plate Visualization
        dbc.Col([
            html.Div([
                html.H1('Welcome to Doser Responser', style={'textAlign': 'center'}),
                html.P('A tool for analyzing dose-response relationships in your experiments.', 
                       style={'textAlign': 'center', 'marginBottom': '30px'}),
            ], className="mb-4"),
            
            html.H2('Upload CSV File', style={'textAlign': 'center'}),
            
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '0 auto 20px',
                },
                multiple=False
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        'Load Sample Data',
                        id='load-sample-button',
                        n_clicks=0,
                        color='secondary',
                        size='sm',
                        style={
                            'width': '100%',
                            'margin': '0 auto 5px',
                            'display': 'block'
                        }
                    ),
                ], width=6),
                dbc.Col([
                    dbc.Button(
                        'Download Sample',
                        id='download-sample-button',
                        n_clicks=0,
                        color='info',
                        size='sm',
                        style={
                            'width': '100%',
                            'margin': '0 auto 5px',
                            'display': 'block'
                        }
                    ),
                ], width=6),
            ], className="mb-2"),
            
            dbc.Button(
                'Calculate Dose Response',
                id='calculate-button',
                n_clicks=0,
                color='primary',
                style={
                    'width': '100%',
                    'margin': '0 auto 20px',
                    'display': 'block'
                }
            ),
            
            # dbc.Button(
            #     'Reset Excluded Wells',
            #     id='reset-excluded-button',
            #     color='secondary',
            #     className='mt-2 mb-3',
            #     size='sm',
            #     style={'width': '100%'}
            # ),
            
            html.P('Click on wells to include/exclude them from the dose response analysis', 
                  style={'textAlign': 'center', 'marginBottom': '20px', 'fontStyle': 'italic'}),
            
            # Plate visualization area
            html.Div(id='output-data-upload'),
            
        ], md=6),
        
        # Right column - Dose Response Curves and Controls
        dbc.Col([
            # Compact header with controls in a single row
            dbc.Row([
                dbc.Col([
                    html.H2('Dose Response Analysis', style={'textAlign': 'left', 'margin': '0'}),
                ], md=6),
                dbc.Col([
                    dbc.Button(
                        "Show Controls", 
                        id="collapse-button",
                        color="secondary",
                        size="sm",
                        n_clicks=0,
                        style={'float': 'right'}
                    ),
                ], md=6),
            ], className="mb-2"),
            
            # Compact controls section - initially collapsed to save space
            dbc.Collapse([
                # Model selection in a compact horizontal layout
                dbc.Row([
                    dbc.Col([
                        html.Label('Model:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=[
                                {'label': '4PL (Hill)', 'value': 'hill'},
                                {'label': '3PL', 'value': '3pl'},
                                {'label': '5PL', 'value': '5pl'},
                                {'label': 'Exponential', 'value': 'exp'},
                            ],
                            value='hill',
                            clearable=False,
                            style={'color': 'black'},
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label('Auto-select best:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dbc.Checkbox(
                            id="use-best-model",
                            label="Use best model (AIC)",
                            value=False,
                        ),
                    ], md=4),
                    dbc.Col([
                        html.Label('Treatments:', style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                        dcc.Dropdown(
                            id='treatment-selector',
                            multi=True,
                            placeholder="Select treatments...",
                            style={'color': 'black'}
                        ),
                    ], md=4),
                ], className="mb-3"),
                
                # Expandable advanced controls
                dbc.Accordion([
                    dbc.AccordionItem([
                        # Parameter sliders in a more compact layout
                        html.Div(id='parameter-sliders'),
                    ], title="Parameter Controls", item_id="params"),
                    
                    dbc.AccordionItem([
                        # Model explanation
                        html.Div(id='model-explanation'),
                    ], title="Model Information", item_id="info"),
                    
                    dbc.AccordionItem([
                        # Model comparison
                        html.Div(id='model-comparison'),
                    ], title="Model Comparison", item_id="comparison"),
                ], id="advanced-controls", start_collapsed=True, className="mb-3"),
                
            ], id="collapse-parameters", is_open=False, className="mb-3"),
            
            # Main graph area - now takes up most of the space
            html.Div([
                # Tabs for different graph views - made more compact
                dbc.Tabs([
                    # Tab 1: Combined graph
                    dbc.Tab([
                        html.Div(id='combined-dose-response-graph')
                    ], label="Combined View", tab_id="tab-combined"),
                    
                    # Tab 2: Individual graphs
                    dbc.Tab([
                        html.Div(id='dose-response-graphs')
                    ], label="Individual Graphs", tab_id="tab-individual"),
                ], id="analysis-tabs", active_tab="tab-combined", className="mb-2"),
            ], style={'height': '70vh', 'overflow': 'auto'}),  # Fixed height with scroll if needed
            
            # Fit statistics (hidden)
            html.Div(id='fit-statistics', style={'display': 'none'}),
        ], md=6)
    ]),
], fluid=True)

# Add download component for sample data
app.layout.children.append(dcc.Download(id="download-sample-csv"))

# Handle file upload and store data
@callback(
    Output('shared-data', 'data'),
    [Input('upload-data', 'contents'),
     Input('load-sample-button', 'n_clicks')],
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def store_uploaded_data(contents, load_sample_clicks, filename):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger == 'load-sample-button' and load_sample_clicks > 0:
        # Load sample data
        try:
            # Get the directory of the current file (dash_app)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to the project root
            project_root = os.path.dirname(current_dir)
            sample_file_path = os.path.join(project_root, 'sample_data.csv')
            
            df = pd.read_csv(sample_file_path)
            return df.to_dict('records')
        except Exception as e:
            print(f"Error loading sample data: {e}")
            return no_update
    
    elif trigger == 'upload-data' and contents is not None:
        # Handle regular file upload
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_dict('records')
        except Exception as e:
            return no_update
    
    return no_update

# Handle sample data download
@callback(
    Output("download-sample-csv", "data"),
    Input("download-sample-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_sample_data(n_clicks):
    if n_clicks:
        # Get the path to the sample data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        sample_file_path = os.path.join(project_root, 'sample_data.csv')
        
        # Return the file for download
        return dcc.send_file(sample_file_path, filename="sample_data.csv")
    return no_update

# Display visualization whenever data changes or excluded wells change
@callback(
    Output('output-data-upload', 'children'),
    [Input('shared-data', 'data'),
     Input('excluded-wells', 'data'),
     Input('expanded-plates', 'data')]
)
def update_visualization(data, excluded_wells, expanded_plates):
    print("Rendering plate with excluded wells:", excluded_wells)
    print("Expanded plates:", expanded_plates)
    
    if data is not None:
        df = pd.DataFrame(data)
        if 'Experiment_ID' not in df.columns:
            return html.Div("CSV file must contain an 'Experiment_ID' column")
            
        experiment_ids = df['Experiment_ID'].unique()
        
        # Create plate icons container
        plate_icons = []
        expanded_plates_content = []
        
        for experiment_id in experiment_ids:
            experiment_df = df[df['Experiment_ID'] == experiment_id]
            plate_type = detect_plate_type(experiment_df)
            
            # Always show the icon
            plate_icon = generate_plate_icon(experiment_df, experiment_id, excluded_wells)
            plate_icons.append(plate_icon)
            
            # Show expanded view if this plate is expanded
            if experiment_id in expanded_plates:
                full_plate = generate_full_plate_visualization(experiment_df, excluded_wells, plate_type)
                expanded_plate_content = html.Div([
                    html.Div([
                        html.H4(f'Experiment ID: {experiment_id}', style={'display': 'inline-block'}),
                        html.Span(f' ({plate_type}-well plate)', style={
                            'color': '#aaa', 
                            'font-size': '14px', 
                            'margin-left': '10px'
                        }),
                        dbc.Button(
                            "Collapse", 
                            id={'type': 'collapse-plate-btn', 'experiment': experiment_id},
                            color="secondary", 
                            size="sm", 
                            style={'float': 'right'}
                        )
                    ]),
                    html.Hr(),
                    full_plate
                ], style={
                    'border': '2px solid #4ecdc4',
                    'border-radius': '8px',
                    'padding': '15px',
                    'margin': '10px 0',
                    'background-color': 'rgba(14, 47, 68, 0.1)'
                })
                expanded_plates_content.append(expanded_plate_content)
        
        # Return both icon view and expanded plates
        return html.Div([
            html.Div([
                html.H3('Plate Overview', style={'textAlign': 'center', 'margin-bottom': '20px'}),
                html.P('Click on plate icons to expand and interact with wells', 
                      style={'textAlign': 'center', 'fontStyle': 'italic', 'color': '#aaa'}),
                html.Div(plate_icons, style={
                    'display': 'flex',
                    'flex-wrap': 'wrap',
                    'justify-content': 'center',
                    'gap': '10px',
                    'margin': '20px 0'
                })
            ]),
            html.Div(expanded_plates_content)
        ])
    
    return html.Div("Upload a CSV file to see plate visualizations")

# Handle well clicks to toggle exclusion
@callback(
    Output('excluded-wells', 'data'),
    Input({'type': 'well-cell', 'well': ALL, 'experiment': ALL}, 'n_clicks'),
    State('excluded-wells', 'data'),
    prevent_initial_call=True
)
def toggle_well_exclusion(n_clicks, excluded_wells):
    ctx = dash.callback_context
    if not ctx.triggered:
        return excluded_wells
    
    # Get the triggered input and check if any n_clicks value is actually > 0
    # This prevents callbacks from firing during initial creation
    if not any(n > 0 for n in n_clicks if n is not None):
        return excluded_wells
    
    # Add debugging print statements
    print("Triggered:", ctx.triggered)
    print("Current excluded wells:", excluded_wells)
    
    # Fix the parsing of the clicked ID
    triggered_id = ctx.triggered[0]['prop_id']
    print("Triggered ID:", triggered_id)
    
    # Extract the JSON part from the string
    match = re.search(r'(\{.*\})', triggered_id)
    if match:
        json_str = match.group(1)
        print("Extracted JSON:", json_str)
        well_info = json.loads(json_str)
        print("Parsed well info:", well_info)
        well = well_info['well']
        print("Selected well:", well)
        
        # Toggle the well's exclusion status
        if well in excluded_wells:
            excluded_wells.remove(well)
        else:
            excluded_wells.append(well)
    else:
        print("No JSON match found in:", triggered_id)
    
    print("New excluded wells:", excluded_wells)
    return excluded_wells

# Handle plate icon clicks to expand/collapse plates
@callback(
    Output('expanded-plates', 'data'),
    [Input({'type': 'plate-icon', 'experiment': ALL}, 'n_clicks'),
     Input({'type': 'collapse-plate-btn', 'experiment': ALL}, 'n_clicks')],
    State('expanded-plates', 'data'),
    prevent_initial_call=True
)
def toggle_plate_expansion(icon_clicks, collapse_clicks, expanded_plates):
    ctx = dash.callback_context
    if not ctx.triggered:
        return expanded_plates
    
    # Get the triggered input
    triggered_id = ctx.triggered[0]['prop_id']
    print("Plate expansion triggered:", triggered_id)
    
    # Extract the JSON part from the string
    match = re.search(r'(\{.*\})', triggered_id)
    if match:
        json_str = match.group(1)
        button_info = json.loads(json_str)
        experiment_id = button_info['experiment']
        button_type = button_info['type']
        
        print(f"Button type: {button_type}, Experiment: {experiment_id}")
        
        if button_type == 'plate-icon':
            # Expand the plate
            if experiment_id not in expanded_plates:
                expanded_plates.append(experiment_id)
        elif button_type == 'collapse-plate-btn':
            # Collapse the plate
            if experiment_id in expanded_plates:
                expanded_plates.remove(experiment_id)
    
    print("New expanded plates:", expanded_plates)
    return expanded_plates

# Add this callback to toggle the collapse state
@callback(
    Output("collapse-parameters", "is_open"),
    Output("collapse-button", "children"),
    Input("collapse-button", "n_clicks"),
    State("collapse-parameters", "is_open"),
)
def toggle_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open, "Hide Controls" if not is_open else "Show Controls"
    return is_open, "Show Controls"

# Show dose response graphs with fitted curves when button is clicked, wells change, or model changes
@callback(
    [Output('calculation-state', 'data'),
     Output('dose-response-graphs', 'children'),
     Output('combined-dose-response-graph', 'children'),
     Output('curve-fit-data', 'data')],
    [Input('calculate-button', 'n_clicks'),
     Input('excluded-wells', 'data'),
     Input('model-dropdown', 'value'),
     Input({'type': 'param-slider', 'param': ALL}, 'value'),
     Input('treatment-selector', 'value'),
     Input('active-tab', 'data')],
    [State('shared-data', 'data'),
     State('calculation-state', 'data'),
     State('curve-fit-data', 'data'),
     State({'type': 'param-slider', 'param': ALL}, 'id')],
    prevent_initial_call=True
)
def handle_calculate_button(n_clicks, excluded_wells, model_type, param_values, 
                           selected_treatments, active_tab, data, calc_state, 
                           curve_fit_data, param_ids):
    # Check what triggered the callback
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Detect parameter slider changes
    param_slider_changed = False
    manual_params = {}
    
    if trigger and 'param-slider' in trigger:
        param_slider_changed = True
        # Extract the slider parameter names and values
        if param_values and param_ids:
            for i, param_id in enumerate(param_ids):
                if i < len(param_values):
                    # Fix: param_id is a dictionary, not a JSON string
                    param_name = param_id['param']
                    manual_params[param_name] = param_values[i]
    
    # Determine what action to take based on the trigger
    update_needed = (trigger == 'calculate-button' and n_clicks > 0) or \
                    (trigger == 'excluded-wells' and calc_state and calc_state.get('show_visualizations', False)) or \
                    (trigger == 'model-dropdown' and calc_state and calc_state.get('show_visualizations', False)) or \
                    param_slider_changed
    
    if update_needed and data is not None:
        # Update calculation state
        new_calc_state = {'show_visualizations': True}
        
        # Generate the dose response graphs with fitted curves
        df = pd.DataFrame(data)
        required_columns = {'Dose_uM', 'Response_Metric', 'Treatment', 'Well'}
        if not required_columns.issubset(df.columns):
            empty_message = html.Div('Required columns are missing in the data.')
            return new_calc_state, empty_message, empty_message, {}

        # Filter out excluded wells
        if excluded_wells:
            df = df[~df['Well'].isin(excluded_wells)]
            
        # Group data by Treatment and Dose_uM to calculate mean and standard deviation
        grouped = df.groupby(['Treatment', 'Dose_uM']).agg(
            mean_response=('Response_Metric', 'mean'),
            std_response=('Response_Metric', 'std')
        ).reset_index()

        # Get unique treatments
        treatments = grouped['Treatment'].unique()
        
        if len(treatments) == 0:
            empty_message = html.Div('No data available after filtering excluded wells.')
            return new_calc_state, empty_message, empty_message, {}
        
        # Store fit results for all treatments
        all_fit_data = curve_fit_data.copy() if curve_fit_data else {}
        
        # Generate individual graphs
        individual_graphs = []
        for treatment in treatments:
            treatment_data = grouped[grouped['Treatment'] == treatment]
            
            # Fit all models for comparison
            model_types = ['hill', '3pl', '5pl', 'exp']
            if treatment not in all_fit_data:
                all_fit_data[treatment] = {}
                
            # Fit all models if this is a fresh calculation (not just parameter adjustment)
            if not param_slider_changed:
                for model in model_types:
                    # Only fit if we don't already have data for this model
                    if model not in all_fit_data[treatment]:
                        x_data = treatment_data['Dose_uM'].values
                        y_data = treatment_data['mean_response'].values
                        fit_result = fit_dose_response_model(x_data, y_data, model)
                        all_fit_data[treatment][model] = fit_result
            
            # Then continue with the rest of the function using the selected model_type
            # Prepare data for plotting
            x_data = treatment_data['Dose_uM'].values
            y_data = treatment_data['mean_response'].values
            error_data = treatment_data['std_response'].values
            
            # Only perform fitting if this is not a slider adjustment or if we don't have fit data yet
            if not param_slider_changed or treatment not in all_fit_data or model_type not in all_fit_data.get(treatment, {}):
                # Fit the selected model to the data
                fit_result = fit_dose_response_model(x_data, y_data, model_type)
                
                # Store the fit result for this treatment
                if treatment not in all_fit_data:
                    all_fit_data[treatment] = {}
                all_fit_data[treatment][model_type] = fit_result
            else:
                # Use existing fit result but with manual parameters if slider was adjusted
                fit_result = all_fit_data[treatment][model_type].copy()
                
                if param_slider_changed and manual_params:
                    # Replace parameters with manual values from sliders
                    original_params = fit_result['params']
                    
                    if model_type == 'hill' and len(original_params) == 4:
                        param_names = ['bottom', 'top', 'ec50', 'hill']
                        for i, name in enumerate(param_names):
                            if name in manual_params:
                                original_params[i] = manual_params[name]
                    
                    elif model_type == '3pl' and len(original_params) == 3:
                        param_names = ['top', 'ec50', 'hill']
                        for i, name in enumerate(param_names):
                            if name in manual_params:
                                original_params[i] = manual_params[name]
                    
                    elif model_type == '5pl' and len(original_params) == 5:
                        param_names = ['bottom', 'top', 'ec50', 'hill', 's']
                        for i, name in enumerate(param_names):
                            if name in manual_params:
                                original_params[i] = manual_params[name]
                    
                    elif model_type == 'exp' and len(original_params) == 3:
                        param_names = ['a', 'b', 'c']
                        for i, name in enumerate(param_names):
                            if name in manual_params:
                                original_params[i] = manual_params[name]
                    
                    fit_result['params'] = original_params
            
            # Create figure
            fig = go.Figure()
            
            # Create combined dataset for plotting with zero handling
            zero_data = treatment_data[treatment_data['Dose_uM'] == 0]
            nonzero_data = treatment_data[treatment_data['Dose_uM'] > 0]
            
            # Find the lowest non-zero dose for zero position
            min_nonzero = nonzero_data['Dose_uM'].min() if not nonzero_data.empty else 0.1
            zero_position = min_nonzero / 10  # Place zero one tick below minimum
            
            # Create combined dataset for plotting
            plot_data = nonzero_data.copy()
            if not zero_data.empty:
                modified_zero = zero_data.copy()
                modified_zero['Dose_uM'] = zero_position
                plot_data = pd.concat([modified_zero, plot_data]).sort_values('Dose_uM')
            
            # Add data points with error bars
            fig.add_trace(go.Scatter(
                x=plot_data['Dose_uM'],
                y=plot_data['mean_response'],
                error_y=dict(type='data', array=plot_data['std_response'], visible=True),
                mode='markers',
                name='Data Points',
                marker=dict(size=10)
            ))
            
            # Add fitted curve if fitting was successful
            if fit_result['success']:
                # Generate smooth x values for curve
                if nonzero_data.empty:
                    x_smooth = np.linspace(0.001, 10, 100)  # Default range if no data
                else:
                    x_min = np.min(nonzero_data['Dose_uM'])
                    x_max = np.max(nonzero_data['Dose_uM'])
                    x_smooth = np.logspace(np.log10(x_min/10), np.log10(x_max*2), 100)
                
                # Calculate y values based on the model type
                params = fit_result['params']
                
                if model_type == 'hill':
                    y_smooth = hill_equation(x_smooth, *params)
                    curve_name = '4PL Fit'
                elif model_type == '3pl':
                    y_smooth = three_param_logistic(x_smooth, *params)
                    curve_name = '3PL Fit'
                elif model_type == '5pl':
                    y_smooth = five_param_logistic(x_smooth, *params)
                    curve_name = '5PL Fit'
                elif model_type == 'exp':
                    y_smooth = exponential_model(x_smooth, *params)
                    curve_name = 'Exponential Fit'
                
                # Add fitted curve
                fig.add_trace(go.Scatter(
                    x=x_smooth,
                    y=y_smooth,
                    mode='lines',
                    name=curve_name,
                    line=dict(color='red', width=2)
                ))
                
                # Add fit statistics to the graph - positioned at top right
                r2 = fit_result['r_squared']
                rmse = fit_result['rmse']
                aic = fit_result.get('aic', 'N/A')
                bic = fit_result.get('bic', 'N/A')

                fig.add_annotation(
                    x=0.98,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=f"R² = {r2:.4f}, RMSE = {rmse:.4f}<br>AIC = {aic:.2f}, BIC = {bic:.2f}",
                    showarrow=False,
                    bgcolor="rgba(14, 47, 68, 0.8)",
                    bordercolor="white",
                    borderwidth=1,
                    borderpad=4,
                    font=dict(color="white")
                )
            
            # Generate tick values to include our zero position
            tick_vals = [zero_position]  # Start with zero position
            
            # Add standard log ticks
            if not nonzero_data.empty:
                decades = range(int(np.floor(np.log10(min_nonzero))), 
                              int(np.ceil(np.log10(nonzero_data['Dose_uM'].max()))) + 1)
                for decade in decades:
                    tick_vals.extend([10**decade])
            
            # Create matching tick labels with "0" for zero_position
            tick_text = ["0"] + [str(val) if val >= 1 else f"{val:.1e}" for val in tick_vals[1:]]
            
            fig.update_layout(
                title=f'Dose Response Curve for {treatment}',
                xaxis_title='Dose (μM)',
                yaxis_title='Response Metric',
                xaxis=dict(
                    type='log',
                    tickmode='array',
                    tickvals=tick_vals,
                    ticktext=tick_text,
                ),
                template='slate',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            individual_graphs.append(dcc.Graph(figure=fig))
        
        # Generate combined graph with selected treatments
        combined_graph = generate_combined_graph(
            grouped, selected_treatments, all_fit_data, model_type
        )
        
        return new_calc_state, individual_graphs, combined_graph, all_fit_data
    
    return no_update, no_update, no_update, no_update

# Define dose-response curve models
def hill_equation(x, bottom, top, ec50, hill):
    """4-parameter logistic model (Hill equation)"""
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)

def three_param_logistic(x, top, ec50, hill):
    """3-parameter logistic model with bottom = 0"""
    return top / (1 + (ec50 / x) ** hill)

def five_param_logistic(x, bottom, top, ec50, hill, s):
    """5-parameter logistic model with asymmetry factor"""
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill) ** s

def exponential_model(x, a, b, c):
    """Simple exponential model"""
    return a * (1 - np.exp(-b * x)) + c

# Function to fit model to data
def fit_dose_response_model(xdata, ydata, model_type, initial_params=None):
    """Fit dose-response data to the specified model"""
    # Handle zero values for log scale
    nonzero_mask = xdata > 0
    xdata_fit = xdata[nonzero_mask]
    ydata_fit = ydata[nonzero_mask]
    
    try:
        if model_type == 'hill':
            # Initial parameter guesses for Hill equation
            if initial_params is None:
                min_y = np.min(ydata_fit)
                max_y = np.max(ydata_fit)
                mid_x = np.median(xdata_fit)
                initial_params = [min_y, max_y, mid_x, 1.0]
            
            params, covariance = curve_fit(
                hill_equation, xdata_fit, ydata_fit, 
                p0=initial_params, 
                bounds=([np.min(ydata_fit)*0.9, np.min(ydata_fit), 0.0001*np.min(xdata_fit), 0.1],
                        [np.max(ydata_fit), np.max(ydata_fit)*1.1, 1000*np.max(xdata_fit), 10]),
                maxfev=10000
            )
            
            # Calculate goodness of fit
            y_pred = hill_equation(xdata_fit, *params)
            
        elif model_type == '3pl':
            if initial_params is None:
                max_y = np.max(ydata_fit)
                mid_x = np.median(xdata_fit)
                initial_params = [max_y, mid_x, 1.0]
            
            params, covariance = curve_fit(
                three_param_logistic, xdata_fit, ydata_fit, 
                p0=initial_params,
                bounds=([0.0, 0.0001*np.min(xdata_fit), 0.1],
                        [np.max(ydata_fit)*1.1, 1000*np.max(xdata_fit), 10]),
                maxfev=10000
            )
            
            # Calculate goodness of fit
            y_pred = three_param_logistic(xdata_fit, *params)
            
        elif model_type == '5pl':
            if initial_params is None:
                min_y = np.min(ydata_fit)
                max_y = np.max(ydata_fit)
                mid_x = np.median(xdata_fit)
                initial_params = [min_y, max_y, mid_x, 1.0, 1.0]
            
            params, covariance = curve_fit(
                five_param_logistic, xdata_fit, ydata_fit, 
                p0=initial_params,
                bounds=([np.min(ydata_fit)*0.9, np.min(ydata_fit), 0.0001*np.min(xdata_fit), 0.1, 0.1],
                        [np.max(ydata_fit), np.max(ydata_fit)*1.1, 1000*np.max(xdata_fit), 10, 10]),
                maxfev=10000
            )
            
            # Calculate goodness of fit
            y_pred = five_param_logistic(xdata_fit, *params)
            
        elif model_type == 'exp':
            if initial_params is None:
                max_y = np.max(ydata_fit)
                initial_params = [max_y, 0.1, np.min(ydata_fit)]
            
            params, covariance = curve_fit(
                exponential_model, xdata_fit, ydata_fit, 
                p0=initial_params,
                maxfev=10000
            )
            
            # Calculate goodness of fit
            y_pred = exponential_model(xdata_fit, *params)
        
        # Calculate R² and RMSE
        residuals = ydata_fit - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata_fit - np.mean(ydata_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Calculate AIC and BIC
        n = len(ydata_fit)  # number of data points
        k = len(params)     # number of parameters
        
        # For normally distributed errors
        aic = n * np.log(ss_res/n) + 2*k
        bic = n * np.log(ss_res/n) + k*np.log(n)
        
        return {
            'params': params.tolist(),  # Convert to list for JSON serialization
            'model_type': model_type,
            'r_squared': r_squared,
            'rmse': rmse,
            'aic': aic,
            'bic': bic,
            'num_params': k,
            'num_points': n,
            'success': True,
        }
    except Exception as e:
        print(f"Curve fitting error: {str(e)}")
        return {
            'params': None,
            'model_type': model_type,
            'r_squared': 0,
            'rmse': float('inf'),
            'aic': float('inf'),
            'bic': float('inf'),
            'success': False,
            'error': str(e)
        }

# Create a helper function for consistent slider creation
def create_slider(param_name, display_name, value, min_val, max_val, step=0.01):
    # Create fewer marks for compact display
    mark_count = 3
    mark_values = np.linspace(min_val, max_val, mark_count)
    marks = {float(mark_values[0]): f"{mark_values[0]:.2f}", 
             float(mark_values[-1]): f"{mark_values[-1]:.2f}"}
    
    return dbc.Row([
        dbc.Col([
            html.Label(f"{display_name}:", style={'font-weight': 'bold', 'margin-bottom': '2px', 'font-size': '12px'}),
        ], width=3),
        dbc.Col([
            dcc.Slider(
                id={'type': 'param-slider', 'param': param_name},
                min=min_val,
                max=max_val,
                value=value,
                marks=marks,
                step=step,
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=9),
    ], className="mb-2")

# Update parameter controls for all model types
@callback(
    [Output('parameter-sliders', 'children'),
     Output('model-comparison', 'children'),
     Output('model-explanation', 'children')],
    [Input('model-dropdown', 'value'),
     Input('curve-fit-data', 'data'),
     Input('active-tab', 'data'),
     Input('treatment-selector', 'value')],
    prevent_initial_call=True
)
def update_parameter_controls(model_type, curve_fit_data, active_tab, selected_treatments):
    # Choose which treatment to use for parameters based on the active tab
    target_treatment = None
    
    if active_tab == "tab-combined" and selected_treatments:
        # For combined view, use the first selected treatment
        target_treatment = selected_treatments[0] if selected_treatments else None
    else:
        # For individual view, just use the first treatment with successful fit
        if curve_fit_data:
            for treatment, models in curve_fit_data.items():
                if model_type in models and models[model_type]['success']:
                    target_treatment = treatment
                    break
    
    # Get model comparison card if treatment data exists
    model_comparison_card = html.Div()
    model_explanation_card = get_model_explanation(model_type)
    
    if target_treatment and curve_fit_data and target_treatment in curve_fit_data:
        model_comparison_card = generate_model_comparison_card(curve_fit_data, target_treatment)
    
    # If we found a treatment with parameters, use them for sliders
    if target_treatment and curve_fit_data and target_treatment in curve_fit_data and \
       model_type in curve_fit_data[target_treatment] and curve_fit_data[target_treatment][model_type]['success']:
        
        fit_params = curve_fit_data[target_treatment][model_type]['params']
        
        # Generate sliders based on the model and fit parameters
        sliders = []
        
        if model_type == 'hill' and len(fit_params) >= 4:
            bottom, top, ec50, hill = fit_params
            sliders = [
                create_slider("bottom", "Bottom", bottom, bottom*0.5, bottom*1.5),
                create_slider("top", "Top", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill", hill, max(0.1, hill*0.5), hill*1.5)
            ]
            
        elif model_type == '3pl' and len(fit_params) >= 3:
            top, ec50, hill = fit_params
            sliders = [
                create_slider("top", "Top", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill", hill, max(0.1, hill*0.5), hill*1.5)
            ]
            
        elif model_type == '5pl' and len(fit_params) >= 5:
            bottom, top, ec50, hill, s = fit_params
            sliders = [
                create_slider("bottom", "Bottom", bottom, bottom*0.5, bottom*1.5),
                create_slider("top", "Top", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill", hill, max(0.1, hill*0.5), hill*1.5),
                create_slider("s", "Asymmetry", s, max(0.1, s*0.5), s*2)
            ]
            
        elif model_type == 'exp' and len(fit_params) >= 3:
            a, b, c = fit_params
            sliders = [
                create_slider("a", "Amplitude", a, a*0.5, a*1.5),
                create_slider("b", "Rate", b, b*0.1, b*10),
                create_slider("c", "Offset", c, c*0.5, c*1.5)
            ]
        
        sliders_div = html.Div([
            html.Div(f"Parameters for: {target_treatment}", 
                    style={"fontStyle": "italic", "marginBottom": "10px", "fontSize": "12px"}),
            html.Div(sliders)
        ])
        
        # Return values matching the updated Output list
        return sliders_div, model_comparison_card, model_explanation_card
    
    # If no fit data is available, return default controls and cards
    return default_parameter_sliders(model_type), model_comparison_card, model_explanation_card

# Function to generate default sliders
def default_parameter_sliders(model_type):
    if model_type == 'hill':
        return html.Div([
            create_slider("bottom", "Bottom", 0, 0, 100),
            create_slider("top", "Top", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill", 1, 0.1, 5)
        ])
    elif model_type == '3pl':
        return html.Div([
            create_slider("top", "Top", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill", 1, 0.1, 5)
        ])
    elif model_type == '5pl':
        return html.Div([
            create_slider("bottom", "Bottom", 0, 0, 100),
            create_slider("top", "Top", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill", 1, 0.1, 5),
            create_slider("s", "Asymmetry", 1, 0.1, 5)
        ])
    elif model_type == 'exp':
        return html.Div([
            create_slider("a", "Amplitude", 100, 1, 200),
            create_slider("b", "Rate", 0.1, 0.001, 1),
            create_slider("c", "Offset", 0, -50, 50)
        ])
    return html.Div()

# Function for parameter explanations
def parameter_explanations(model_type, fit_data=None, treatment=None):
    # Create base explanations
    model_explanation = get_model_explanation(model_type)
    
    if fit_data and treatment and treatment in fit_data and model_type in fit_data[treatment]:
        # Add model comparison card first
        model_comparison = generate_model_comparison_card(fit_data, treatment)
        
        # Return just the model comparison and base explanation
        return html.Div([model_comparison, model_explanation])
    
    return html.Div([model_explanation])

def get_model_explanation(model_type):
    # Compact model explanations
    if model_type == 'hill':
        return dbc.Card(
            dbc.CardBody([
                html.H5("4-Parameter Logistic (Hill)", className="card-title"),
                html.P("y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)", 
                       style={'font-family': 'monospace', 'font-size': '12px'}),
                html.Ul([
                    html.Li("Bottom: Lower asymptote"),
                    html.Li("Top: Upper asymptote"), 
                    html.Li("EC50: Half-maximal concentration"),
                    html.Li("Hill: Slope factor")
                ], style={'font-size': '12px', 'margin': '0'})
            ], style={'padding': '10px'}), 
            style={"background-color": "rgba(14, 47, 68, 0.3)"}
        )
    
    elif model_type == '3pl':
        return dbc.Card(
            dbc.CardBody([
                html.H5("3-Parameter Logistic", className="card-title"),
                html.P("y = Top / (1 + (EC50/x)^Hill)", 
                       style={'font-family': 'monospace', 'font-size': '12px'}),
                html.P("Lower asymptote fixed at zero.", style={'font-size': '12px', 'margin': '5px 0'}),
                html.Ul([
                    html.Li("Top: Upper asymptote"),
                    html.Li("EC50: Half-maximal concentration"),
                    html.Li("Hill: Slope factor")
                ], style={'font-size': '12px', 'margin': '0'})
            ], style={'padding': '10px'}), 
            style={"background-color": "rgba(14, 47, 68, 0.3)"}
        )
    
    elif model_type == '5pl':
        return dbc.Card(
            dbc.CardBody([
                html.H5("5-Parameter Logistic", className="card-title"),
                html.P("y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^s", 
                       style={'font-family': 'monospace', 'font-size': '12px'}),
                html.Ul([
                    html.Li("s: Asymmetry factor (s=1 → 4PL)")
                ], style={'font-size': '12px', 'margin': '0'})
            ], style={'padding': '10px'}), 
            style={"background-color": "rgba(14, 47, 68, 0.3)"}
        )
    
    elif model_type == 'exp':
        return dbc.Card(
            dbc.CardBody([
                html.H5("Exponential Model", className="card-title"),
                html.P("y = a * (1 - e^(-b*x)) + c", 
                       style={'font-family': 'monospace', 'font-size': '12px'}),
                html.Ul([
                    html.Li("a: Amplitude"),
                    html.Li("b: Rate constant"),
                    html.Li("c: Offset")
                ], style={'font-size': '12px', 'margin': '0'})
            ], style={'padding': '10px'}), 
            style={"background-color": "rgba(14, 47, 68, 0.3)"}
        )
    
    return html.Div()

# Populate treatment selector dropdown when data is available
@callback(
    Output('treatment-selector', 'options'),
    Output('treatment-selector', 'value'),
    Input('shared-data', 'data'),
    prevent_initial_call=True
)
def update_treatment_selector(data):
    if data:  # Remove dependency on calc_state
        df = pd.DataFrame(data)
        treatments = df['Treatment'].unique().tolist()
        options = [{'label': t, 'value': t} for t in treatments]
        # Select the first treatment by default
        default_selection = [treatments[0]] if treatments else []
        return options, default_selection
    return [], []

# Track active tab
@callback(
    Output('active-tab', 'data'),
    Input('analysis-tabs', 'active_tab'),
    prevent_initial_call=True
)
def update_active_tab(active_tab):
    return active_tab

def generate_combined_graph(grouped_data, selected_treatments, fit_data, model_type):
    """Generate a single graph with multiple selected treatments"""
    if not selected_treatments:
        return html.Div("Select treatments to display", style={'textAlign': 'center', 'margin': '40px'})
    
    # Create a single figure for all selected treatments
    fig = go.Figure()
    
    # Define a color scale with distinct colors for different treatments
    colors = px.colors.qualitative.Plotly  # Built-in distinct color sequence
    
    for i, treatment in enumerate(selected_treatments):
        # Get color for this treatment
        color = colors[i % len(colors)]
        
        # Filter data for this treatment
        treatment_data = grouped_data[grouped_data['Treatment'] == treatment]
        
        if treatment_data.empty:
            continue
            
        # Handle zero doses for log scale
        zero_data = treatment_data[treatment_data['Dose_uM'] == 0]
        nonzero_data = treatment_data[treatment_data['Dose_uM'] > 0]
        
        # Find the lowest non-zero dose for zero position
        min_nonzero = nonzero_data['Dose_uM'].min() if not nonzero_data.empty else 0.1
        zero_position = min_nonzero / 10
        
        # Create combined dataset for plotting
        plot_data = nonzero_data.copy()
        if not zero_data.empty:
            modified_zero = zero_data.copy()
            modified_zero['Dose_uM'] = zero_position
            plot_data = pd.concat([modified_zero, plot_data]).sort_values('Dose_uM')
        
        # Add data points with error bars
        fig.add_trace(go.Scatter(
            x=plot_data['Dose_uM'],
            y=plot_data['mean_response'],
            error_y=dict(type='data', array=plot_data['std_response'], visible=True),
            mode='markers',
            name=f'{treatment} data',
            marker=dict(size=10, color=color)
        ))
        
        # Add fitted curve if available
        if (treatment in fit_data and 
            model_type in fit_data[treatment] and 
            fit_data[treatment][model_type]['success']):
            
            fit_result = fit_data[treatment][model_type]
            params = fit_result['params']
            
            # Generate smooth x values for curve
            if nonzero_data.empty:
                x_smooth = np.linspace(0.001, 10, 100)
            else:
                x_min = np.min(nonzero_data['Dose_uM'])
                x_max = np.max(nonzero_data['Dose_uM'])
                x_smooth = np.logspace(np.log10(x_min/10), np.log10(x_max*2), 100)
            
            # Calculate y values based on the model
            if model_type == 'hill':
                y_smooth = hill_equation(x_smooth, *params)
            elif model_type == '3pl':
                y_smooth = three_param_logistic(x_smooth, *params)
            elif model_type == '5pl':
                y_smooth = five_param_logistic(x_smooth, *params)
            elif model_type == 'exp':
                y_smooth = exponential_model(x_smooth, *params)
            
            # Add fitted curve with matching color
            fig.add_trace(go.Scatter(
                x=x_smooth,
                y=y_smooth,
                mode='lines',
                name=f'{treatment} fit',
                line=dict(color=color, width=2, dash='dash')
            ))
    
    # Create x-axis with proper zero handling
    tick_vals = []
    tick_text = []
    
    # Find overall min non-zero dose across all selected treatments
    all_nonzero = grouped_data[
        (grouped_data['Treatment'].isin(selected_treatments)) & 
        (grouped_data['Dose_uM'] > 0)
    ]['Dose_uM']
    
    if not all_nonzero.empty:
        min_nonzero = all_nonzero.min()
        max_nonzero = all_nonzero.max()
        zero_position = min_nonzero / 10
        
        # Add zero position
        tick_vals.append(zero_position)
        tick_text.append("0")
        
        # Add standard log ticks
        decades = range(
            int(np.floor(np.log10(min_nonzero))),
            int(np.ceil(np.log10(max_nonzero))) + 1
        )
        for decade in decades:
            tick_vals.append(10**decade)
            tick_text.append(str(10**decade) if 10**decade >= 1 else f"{10**decade:.1e}")
    
    # Add a single annotation with all stats
    stats_text = []
    for i, treatment in enumerate(selected_treatments):
        if (treatment in fit_data and 
            model_type in fit_data[treatment] and 
            fit_data[treatment][model_type]['success']):
            
            fit_result = fit_data[treatment][model_type]
            r2 = fit_result['r_squared']
            rmse = fit_result['rmse']
            color_span = f'<span style="color:{colors[i % len(colors)]}">■</span>'
            stats_text.append(f"{color_span} {treatment}: R² = {r2:.4f}, RMSE = {rmse:.4f}")

    if stats_text:
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text="<br>".join(stats_text),
            align="right",
            showarrow=False,
            bgcolor="rgba(14, 47, 68, 0.9)",
            bordercolor="white",
            borderwidth=1,
            borderpad=6,
            font=dict(color="white", size=10),
            width=220,  # Fixed width
            height=20 * len(stats_text) + 10  # Dynamic height based on number of treatments
        )
    
    # Update layout for the combined graph
    fig.update_layout(
        title='Combined Dose Response Analysis',
        xaxis_title='Dose (μM)',
        yaxis_title='Response Metric',
        xaxis=dict(
            type='log',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        template='slate',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dcc.Graph(figure=fig)

def generate_individual_graphs(grouped_data, treatments, fit_data, model_type):
    """Generate tabs for individual treatment graphs with their own controls"""
    
    if not treatments.size:
        return html.Div("No data available to display", 
                       style={'textAlign': 'center', 'margin': '40px'})
    
    # Create a tab for each treatment
    treatment_tabs = []
    
    for i, treatment in enumerate(treatments):
        treatment_data = grouped_data[grouped_data['Treatment'] == treatment]
        
        # Create graph for this treatment (similar to before)
        fig = create_treatment_graph(treatment_data, treatment, fit_data, model_type)
        
        # Create parameter sliders specific to this treatment if fit exists
        param_sliders = create_treatment_param_sliders(treatment, fit_data, model_type)
        
        # Create tab with graph and controls
        tab_content = html.Div([
            dcc.Graph(figure=fig),
            html.Hr(),
            html.Div([
                html.H5("Adjust Model Parameters"),
                param_sliders
            ], id={'type': 'treatment-params', 'treatment': treatment})
        ])
        
        treatment_tabs.append(dbc.Tab(tab_content, label=treatment, tab_id=f"tab-{treatment}"))
    
    return dbc.Tabs(treatment_tabs, id="treatment-tabs", active_tab=f"tab-{treatments[0]}")

def create_treatment_graph(treatment_data, treatment, fit_data, model_type):
    """Create a dose response graph for a single treatment"""
    # Create figure
    fig = go.Figure()
    
    # Extract data for plotting
    x_data = treatment_data['Dose_uM'].values
    y_data = treatment_data['mean_response'].values
    error_data = treatment_data['std_response'].values
    
    # Handle zero doses for log scale
    zero_data = treatment_data[treatment_data['Dose_uM'] == 0]
    nonzero_data = treatment_data[treatment_data['Dose_uM'] > 0]
    
    # Find the lowest non-zero dose for zero position
    min_nonzero = nonzero_data['Dose_uM'].min() if not nonzero_data.empty else 0.1
    zero_position = min_nonzero / 10
    
    # Create combined dataset for plotting
    plot_data = nonzero_data.copy()
    if not zero_data.empty:
        modified_zero = zero_data.copy()
        modified_zero['Dose_uM'] = zero_position
        plot_data = pd.concat([modified_zero, plot_data]).sort_values('Dose_uM')
    
    # Add data points with error bars
    fig.add_trace(go.Scatter(
        x=plot_data['Dose_uM'],
        y=plot_data['mean_response'],
        error_y=dict(type='data', array=plot_data['std_response'], visible=True),
        mode='markers',
        name='Data Points',
        marker=dict(size=10)
    ))
    
    # Add fitted curve if available
    if (treatment in fit_data and 
        model_type in fit_data[treatment] and 
        fit_data[treatment][model_type]['success']):
        
        fit_result = fit_data[treatment][model_type]
        params = fit_result['params']
        
        # Generate smooth x values for curve
        if nonzero_data.empty:
            x_smooth = np.linspace(0.001, 10, 100)
        else:
            x_min = np.min(nonzero_data['Dose_uM'])
            x_max = np.max(nonzero_data['Dose_uM'])
            x_smooth = np.logspace(np.log10(x_min/10), np.log10(x_max*2), 100)
        
        # Calculate y values based on the model
        if model_type == 'hill':
            y_smooth = hill_equation(x_smooth, *params)
            curve_name = '4PL Fit'
        elif model_type == '3pl':
            y_smooth = three_param_logistic(x_smooth, *params)
            curve_name = '3PL Fit'
        elif model_type == '5pl':
            y_smooth = five_param_logistic(x_smooth, *params)
            curve_name = '5PL Fit'
        elif model_type == 'exp':
            y_smooth = exponential_model(x_smooth, *params)
            curve_name = 'Exponential Fit'
        
        # Add fitted curve
        fig.add_trace(go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode='lines',
            name=curve_name,
            line=dict(color='red', width=2)
        ))
        
        # Add fit statistics to the graph - positioned at top right
        r2 = fit_result['r_squared']
        rmse = fit_result['rmse']
        aic = fit_result.get('aic', 'N/A')
        bic = fit_result.get('bic', 'N/A')

        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"R² = {r2:.4f}, RMSE = {rmse:.4f}<br>AIC = {aic:.2f}, BIC = {bic:.2f}",
            showarrow=False,
            bgcolor="rgba(14, 47, 68, 0.8)",
            bordercolor="white",
            borderwidth=1,
            borderpad=4,
            font=dict(color="white")
        )
    
    # Generate tick values to include our zero position
    tick_vals = [zero_position]  # Start with zero position
    
    # Add standard log ticks
    if not nonzero_data.empty:
        decades = range(int(np.floor(np.log10(min_nonzero))), 
                       int(np.ceil(np.log10(nonzero_data['Dose_uM'].max()))) + 1)
        for decade in decades:
            tick_vals.extend([10**decade])
    
    # Create matching tick labels with "0" for zero_position
    tick_text = ["0"] + [str(val) if val >= 1 else f"{val:.1e}" for val in tick_vals[1:]]
    
    # Update layout with titles and template
    fig.update_layout(
        title=f'Dose Response Curve for {treatment}',
        xaxis_title='Dose (μM)',
        yaxis_title='Response Metric',
        xaxis=dict(
            type='log',
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
        ),
        template='slate',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_treatment_param_sliders(treatment, fit_data, model_type):
    """Create parameter sliders specific to a treatment's fit results"""
    # Check if we have fit data for this treatment and model
    if not fit_data or treatment not in fit_data or model_type not in fit_data[treatment]:
        return html.Div("No fit parameters available for this treatment")
    
    fit_result = fit_data[treatment][model_type]
    
    # If fit was not successful, show message
    if not fit_result['success']:
        return html.Div(f"Curve fitting failed: {fit_result.get('error', 'Unknown error')}")
    
    params = fit_result['params']
    sliders = []
    
    # Generate sliders based on the model type
    if model_type == 'hill' and len(params) >= 4:
        bottom, top, ec50, hill = params
        sliders = [
            create_slider(f"{treatment}-bottom", "Bottom Asymptote", bottom, bottom*0.5, bottom*1.5),
            create_slider(f"{treatment}-top", "Top Asymptote", top, top*0.5, top*1.5),
            create_slider(f"{treatment}-ec50", "EC50", ec50, ec50*0.1, ec50*10),
            create_slider(f"{treatment}-hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5)
        ]
    
    elif model_type == '3pl' and len(params) >= 3:
        top, ec50, hill = params
        sliders = [
            create_slider(f"{treatment}-top", "Top Asymptote", top, top*0.5, top*1.5),
            create_slider(f"{treatment}-ec50", "EC50", ec50, ec50*0.1, ec50*10),
            create_slider(f"{treatment}-hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5)
        ]
    
    elif model_type == '5pl' and len(params) >= 5:
        bottom, top, ec50, hill, s = params
        sliders = [
            create_slider(f"{treatment}-bottom", "Bottom Asymptote", bottom, bottom*0.5, bottom*1.5),
            create_slider(f"{treatment}-top", "Top Asymptote", top, top*0.5, top*1.5),
            create_slider(f"{treatment}-ec50", "EC50", ec50, ec50*0.1, ec50*10),
            create_slider(f"{treatment}-hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5),
            create_slider(f"{treatment}-s", "Asymmetry Factor", s, max(0.1, s*0.5), s*2)
        ]
    
    elif model_type == 'exp' and len(params) >= 3:
        a, b, c = params
        sliders = [
            create_slider(f"{treatment}-a", "Amplitude", a, a*0.5, a*1.5),
            create_slider(f"{treatment}-b", "Rate Constant", b, b*0.1, b*10),
            create_slider(f"{treatment}-c", "Offset", c, c*0.5, c*1.5)
        ]
    
    return html.Div(sliders)

def get_best_model(fit_data, treatment):
    """Compare all models for a treatment and determine best fit based on AIC/BIC"""
    if not fit_data or treatment not in fit_data:
        return None
    
    models = ['hill', '3pl', '5pl', 'exp']
    comparison = []
    
    for model in models:
        if model in fit_data[treatment] and fit_data[treatment][model]['success']:
            result = fit_data[treatment][model]
            comparison.append({
                'model': model,
                'aic': result['aic'],
                'bic': result['bic'],
                'r_squared': result['r_squared'],
                'rmse': result['rmse'],
                'num_params': result['num_params'],
                'display_name': {
                    'hill': '4-Parameter Logistic',
                    '3pl': '3-Parameter Logistic',
                    '5pl': '5-Parameter Logistic', 
                    'exp': 'Exponential'
                }[model]
            })
    
    # Sort by AIC (lowest is best)
    comparison_sorted = sorted(comparison, key=lambda x: x['aic'])
    return comparison_sorted

def generate_model_comparison_card(fit_data, treatment):
    """Generate a card showing model comparison"""
    if not fit_data or treatment not in fit_data:
        return html.Div("No model data available")
    
    comparison = get_best_model(fit_data, treatment)
    if not comparison:
        return html.Div("No successful model fits available")
    
    # Create the comparison table
    header = html.Tr([
        html.Th("Model"),
        html.Th("AIC"),
        html.Th("BIC"),
        html.Th("R²"),
        html.Th("Parameters")
    ])
    
    rows = []
    best_aic_model = comparison[0]['model']
    best_bic_model = sorted(comparison, key=lambda x: x['bic'])[0]['model']
    
    for model_data in comparison:
        is_best_aic = model_data['model'] == best_aic_model
        is_best_bic = model_data['model'] == best_bic_model
        
        # Set text color to light green if it's the best model by AIC or BIC
        style = {}
        if is_best_aic or is_best_bic:
            style = {'color': 'lightgreen'}
        
        # Create table row with the style attribute applied to the row
        row = html.Tr([
            html.Td(model_data['display_name']),
            html.Td(f"{model_data['aic']:.2f}"),
            html.Td(f"{model_data['bic']:.2f}"),
            html.Td(f"{model_data['r_squared']:.4f}"),
            html.Td(f"{model_data['num_params']}")
        ], style=style)
        
        rows.append(row)
    
    table = html.Table([header] + rows, className="table table-sm", style={
        'width': '100%',
        'border-collapse': 'collapse',
        'margin-top': '10px',
        'color': 'white'
    })
    
    return dbc.Card(
        dbc.CardBody([
            html.H4("Model Comparison", className="card-title"),
            html.Hr(),
            table,
            html.Hr(),
            html.P([
                html.Strong("Recommendation: "), 
                "The model with the lowest AIC/BIC provides the best balance between fit quality and model complexity."
            ])
        ]),
        className="mb-3",
        style={"background-color": "rgba(14, 47, 68, 0.3)"}
    )

@callback(
    Output('model-dropdown', 'value'),
    [Input('use-best-model', 'value'),
     Input('treatment-selector', 'value')],
    [State('curve-fit-data', 'data')],
    prevent_initial_call=True
)
def update_to_best_model(use_best, selected_treatments, fit_data):
    if not use_best or not fit_data or not selected_treatments:
        return no_update
    
    treatment = selected_treatments[0] if selected_treatments else None
    if not treatment or treatment not in fit_data:
        return no_update
    
    comparison = get_best_model(fit_data, treatment)
    if comparison:
        return comparison[0]['model']  # Return best model by AIC
    
    return no_update

if __name__ == '__main__':
    app.run_server(debug=True)