# index.py

import dash
from dash import html, dcc, Input, Output, State, callback, no_update, MATCH, ALL
import dash_bootstrap_components as dbc
from dash_app.app import app, server
import pandas as pd
import numpy as np
import base64
import io
import plotly.graph_objs as go
import plotly.express as px
from dash_bootstrap_templates import load_figure_template
import json
from scipy.optimize import curve_fit

# Load the 'vapor' theme
load_figure_template('vapor')

# Define the plate visualization function directly in index.py
def generate_plate_visualization(experiment_df, excluded_wells=None):
    if excluded_wells is None:
        excluded_wells = []
        
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
                # Check if this well is excluded
                is_excluded = well in excluded_wells
                
                # Create cell content with text decoration if excluded
                text_style = {'text-decoration': 'line-through'} if is_excluded else {}
                
                # Create a more compact cell content layout
                cell_content = html.Div([
                    html.Div(f'{cell_data.iloc[0]["Treatment"]}', 
                             style={'font-weight': 'bold', 'margin-bottom': '1px', **text_style}),
                    html.Div(f'{cell_data.iloc[0]["Dose_uM"]}μM', 
                             style={'font-size': '8px', 'margin-bottom': '1px', **text_style}),
                    html.Div(f'{cell_data.iloc[0]["Response_Metric"]:.2f}', 
                             style={'font-size': '8px', **text_style}),
                ], style={'font-size': '9px', 'text-align': 'center', 'line-height': '1.0'})
                
                # Increase cell size to accommodate content better
                cell_style = {
                    'border': '1px solid black',
                    'width': '70px',  # Increased width
                    'height': '65px', # Increased height
                    'background-color': '#0a1d2a' if is_excluded else '#0e2f44',
                    'cursor': 'pointer',  # Change cursor to indicate clickable
                    'padding': '2px'   # Add a bit of padding
                }
                
                # Add a unique ID for this cell to track clicks
                cell_id = {'type': 'well-cell', 'well': well, 'experiment': cell_data.iloc[0]['Experiment_ID']}
            else:
                cell_content = html.Div()
                cell_style = {
                    'border': '1px solid black',
                    'width': '60px',
                    'height': '60px',
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

# Simplified navbar without page navigation
navbar = dbc.NavbarSimple(
    brand="Doser Responser",
    brand_href="/",
    color="primary",
    dark=True,
)

# Create the merged single-page layout
app.layout = dbc.Container([
    navbar,
    
    # Stores for data persistence
    dcc.Store(id='shared-data', storage_type='memory'),
    dcc.Store(id='calculation-state', storage_type='memory', data={'show_visualizations': False}),
    dcc.Store(id='excluded-wells', storage_type='memory', data=[]),  # Store for excluded wells
    dcc.Store(id='curve-fit-data', storage_type='memory', data={}),  # Store for curve fitting results
    dcc.Store(id='active-tab', storage_type='memory', data="tab-combined"),
    
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
            
            html.P('Click on wells to include/exclude them from the dose response analysis', 
                  style={'textAlign': 'center', 'marginBottom': '20px', 'fontStyle': 'italic'}),
            
            # Plate visualization area
            html.Div(id='output-data-upload'),
            
        ], md=6),
        
        # Right column - Dose Response Curves and Controls
        dbc.Col([
            html.H1('Dose Response Analysis', style={'textAlign': 'center'}),
            
            # Common controls section - visible for both tabs
            html.Div([
                html.Div([
                    html.Label('Select Curve Model:'),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[
                            {'label': '4-Parameter Logistic (Hill Equation)', 'value': 'hill'},
                            {'label': '3-Parameter Logistic', 'value': '3pl'},
                            {'label': '5-Parameter Logistic', 'value': '5pl'},
                            {'label': 'Exponential', 'value': 'exp'},
                        ],
                        value='hill',
                        clearable=False
                    ),
                ], style={'margin': '10px 0 20px 0'}),
                
                # Parameter sliders and explanations - will be dynamically populated
                dbc.Row([
                    dbc.Col([
                        html.Div(id='parameter-sliders'),
                    ], md=6),
                    dbc.Col([
                        html.Div(id='parameter-explanations', style={'margin': '0 0 20px 0'}),
                    ], md=6),
                ]),
            ], className="p-4 mb-2", style={'background': 'rgba(14, 47, 68, 0.1)', 'border-radius': '5px'}),
            
            # Tabs for different graph views
            dbc.Tabs([
                # Tab 1: Combined graph with multi-drug selection
                dbc.Tab([
                    html.Div([
                        html.Label('Select Treatments to Display:'),
                        dcc.Dropdown(
                            id='treatment-selector',
                            multi=True,
                            style={'margin-bottom': '20px'}
                        ),
                        html.Div(id='combined-dose-response-graph')
                    ], className="p-4"),
                ], label="Combined View", tab_id="tab-combined"),
                
                # Tab 2: Individual graphs with separate controls
                dbc.Tab([
                    html.Div([
                        # Individual graphs area (sliders moved to common area above)
                        html.Div(id='dose-response-graphs'),
                    ], className="p-4"),
                ], label="Individual Graphs", tab_id="tab-individual"),
            ], id="analysis-tabs", active_tab="tab-combined"),
            
            # Fit statistics (hidden)
            html.Div(id='fit-statistics', style={'display': 'none'}),
        ], md=6)
    ]),
], fluid=True)

# Handle file upload and store data
@callback(
    Output('shared-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def store_uploaded_data(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df.to_dict('records')
        except Exception as e:
            return no_update
    return no_update

# Display visualization whenever data changes or excluded wells change
@callback(
    Output('output-data-upload', 'children'),
    [Input('shared-data', 'data'),
     Input('excluded-wells', 'data')]
)
def update_visualization(data, excluded_wells):
    if data is not None:
        df = pd.DataFrame(data)
        if 'Experiment_ID' not in df.columns:
            return html.Div("CSV file must contain an 'Experiment_ID' column")
            
        experiment_ids = df['Experiment_ID'].unique()
        display_visualizations = []
        
        for experiment_id in experiment_ids:
            experiment_df = df[df['Experiment_ID'] == experiment_id]
            plate_visual = generate_plate_visualization(experiment_df, excluded_wells)
            display_visualizations.append(html.Div([
                html.H3(f'Experiment ID: {experiment_id}'),
                plate_visual
            ], style={'margin-bottom': '40px'}))
            
        return display_visualizations
    return html.Div("Upload a CSV file to see plate visualizations")

# Handle well clicks to toggle exclusion
@callback(
    Output('excluded-wells', 'data'),
    Input({'type': 'well-cell', 'well': ALL, 'experiment': ALL}, 'n_clicks'),
    State('excluded-wells', 'data'),
    prevent_initial_call=True
)
def toggle_well_exclusion(n_clicks, excluded_wells):
    # Find which well was clicked
    ctx = dash.callback_context
    if not ctx.triggered:
        return excluded_wells
    
    # Get the ID of the clicked well
    clicked_id = ctx.triggered[0]['prop_id'].split('.')[0]
    well_info = json.loads(clicked_id)
    well = well_info['well']
    
    # Toggle the well's exclusion status
    if well in excluded_wells:
        excluded_wells.remove(well)
    else:
        excluded_wells.append(well)
    
    return excluded_wells

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
        
        # Generate individual graphs - similar to before
        individual_graphs = []
        for treatment in treatments:
            treatment_data = grouped[grouped['Treatment'] == treatment]
            
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
                fig.add_annotation(
                    x=0.98,
                    y=0.98,  # Changed from 0.05 to 0.98 to move to top right
                    xref="paper",
                    yref="paper",
                    text=f"R² = {r2:.4f}, RMSE = {rmse:.4f}",
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
                template='vapor',
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
        
        return {
            'params': params.tolist(),  # Convert to list for JSON serialization
            'model_type': model_type,
            'r_squared': r_squared,
            'rmse': rmse,
            'success': True,
        }
    except Exception as e:
        print(f"Curve fitting error: {str(e)}")
        return {
            'params': None,
            'model_type': model_type,
            'r_squared': 0,
            'rmse': float('inf'),
            'success': False,
            'error': str(e)
        }

# Create a helper function for consistent slider creation
def create_slider(param_name, display_name, value, min_val, max_val, step=0.01):
    # Create evenly distributed marks
    mark_count = 5
    mark_values = np.linspace(min_val, max_val, mark_count)
    marks = {float(val): f"{val:.2f}" for val in mark_values}
    
    return html.Div([
        html.Label(f"{display_name}:", 
                   style={'font-weight': 'bold', 'margin-bottom': '5px'}),
        dcc.Slider(
            id={'type': 'param-slider', 'param': param_name},
            min=min_val,
            max=max_val,
            value=value,
            marks=marks,
            step=step,
            tooltip={"placement": "bottom", "always_visible": True}
        )
    ], style={'margin': '20px 0', 'padding': '10px', 'background': 'rgba(14, 47, 68, 0.2)', 'border-radius': '5px'})

# Update parameter controls for all model types
@callback(
    [Output('parameter-sliders', 'children'),
     Output('parameter-explanations', 'children')],
    [Input('model-dropdown', 'value'),
     Input('curve-fit-data', 'data'),
     Input('active-tab', 'data'),
     Input('treatment-selector', 'value')],
    prevent_initial_call=True
)
def update_parameter_controls(model_type, curve_fit_data, active_tab, selected_treatments):
    """Create sliders based on the selected model and fitted parameters"""
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
    
    # If we found a treatment with parameters, use them
    if target_treatment and curve_fit_data and target_treatment in curve_fit_data and \
       model_type in curve_fit_data[target_treatment] and curve_fit_data[target_treatment][model_type]['success']:
        
        fit_params = curve_fit_data[target_treatment][model_type]['params']
        
        # Generate sliders based on the model and fit parameters
        sliders = []
        
        if model_type == 'hill' and len(fit_params) >= 4:
            bottom, top, ec50, hill = fit_params
            sliders = [
                create_slider("bottom", "Bottom Asymptote", bottom, bottom*0.5, bottom*1.5),
                create_slider("top", "Top Asymptote", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5)
            ]
            
        elif model_type == '3pl' and len(fit_params) >= 3:
            top, ec50, hill = fit_params
            sliders = [
                create_slider("top", "Top Asymptote", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5)
            ]
            
        elif model_type == '5pl' and len(fit_params) >= 5:
            bottom, top, ec50, hill, s = fit_params
            sliders = [
                create_slider("bottom", "Bottom Asymptote", bottom, bottom*0.5, bottom*1.5),
                create_slider("top", "Top Asymptote", top, top*0.5, top*1.5),
                create_slider("ec50", "EC50", ec50, ec50*0.1, ec50*10),
                create_slider("hill", "Hill Slope", hill, max(0.1, hill*0.5), hill*1.5),
                create_slider("s", "Asymmetry Factor", s, max(0.1, s*0.5), s*2)
            ]
            
        elif model_type == 'exp' and len(fit_params) >= 3:
            a, b, c = fit_params
            sliders = [
                create_slider("a", "Amplitude", a, a*0.5, a*1.5),
                create_slider("b", "Rate Constant", b, b*0.1, b*10),
                create_slider("c", "Offset", c, c*0.5, c*1.5)
            ]
        
        return html.Div([
            html.Div(f"Parameters shown for: {target_treatment}", 
                    style={"fontStyle": "italic", "marginBottom": "10px"}),
            html.Div(sliders)
        ]), parameter_explanations(model_type)
    
    # If no fit data is available, return default controls
    return default_parameter_sliders(model_type), parameter_explanations(model_type)

# Function to generate default sliders
def default_parameter_sliders(model_type):
    if model_type == 'hill':
        return html.Div([
            create_slider("bottom", "Bottom Asymptote", 0, 0, 100),
            create_slider("top", "Top Asymptote", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill Slope", 1, 0.1, 5)
        ])
    elif model_type == '3pl':
        return html.Div([
            create_slider("top", "Top Asymptote", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill Slope", 1, 0.1, 5)
        ])
    elif model_type == '5pl':
        return html.Div([
            create_slider("bottom", "Bottom Asymptote", 0, 0, 100),
            create_slider("top", "Top Asymptote", 100, 1, 200),
            create_slider("ec50", "EC50", 1, 0.001, 100),
            create_slider("hill", "Hill Slope", 1, 0.1, 5),
            create_slider("s", "Asymmetry Factor", 1, 0.1, 5)
        ])
    elif model_type == 'exp':
        return html.Div([
            create_slider("a", "Amplitude", 100, 1, 200),
            create_slider("b", "Rate Constant", 0.1, 0.001, 1),
            create_slider("c", "Offset", 0, -50, 50)
        ])
    return html.Div()

# Function for parameter explanations
def parameter_explanations(model_type):
    if model_type == 'hill':
        return html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("4-Parameter Logistic Model", className="card-title"),
                    html.P("Formula: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)"),
                    html.Hr(),
                    html.Ul([
                        html.Li([html.B("Bottom: "), "Lower asymptote (response at zero dose)"]),
                        html.Li([html.B("Top: "), "Upper asymptote (maximum response at infinite dose)"]),
                        html.Li([html.B("EC50: "), "Concentration producing 50% of maximum response"]),
                        html.Li([html.B("Hill: "), "Slope factor (steepness of the curve)"])
                    ], style={'text-align': 'left'})
                ]), 
                className="mb-3", 
                style={"background-color": "rgba(14, 47, 68, 0.3)"}
            )
        ])
    
    elif model_type == '3pl':
        return html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("3-Parameter Logistic Model", className="card-title"),
                    html.P("Formula: y = Top / (1 + (EC50/x)^Hill)"),
                    html.Hr(),
                    html.P("This model assumes the lower asymptote is fixed at zero."),
                    html.Ul([
                        html.Li([html.B("Top: "), "Upper asymptote (maximum response)"]),
                        html.Li([html.B("EC50: "), "Concentration producing 50% of maximum response"]),
                        html.Li([html.B("Hill: "), "Slope factor (steepness of the curve)"])
                    ], style={'text-align': 'left'})
                ]), 
                className="mb-3", 
                style={"background-color": "rgba(14, 47, 68, 0.3)"}
            )
        ])
    
    elif model_type == '5pl':
        return html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("5-Parameter Logistic Model", className="card-title"),
                    html.P("Formula: y = Bottom + (Top - Bottom) / (1 + (EC50/x)^Hill)^s"),
                    html.Hr(),
                    html.Ul([
                        html.Li([html.B("Bottom: "), "Lower asymptote (response at zero dose)"]),
                        html.Li([html.B("Top: "), "Upper asymptote (maximum response)"]),
                        html.Li([html.B("EC50: "), "Concentration producing 50% of maximum response"]),
                        html.Li([html.B("Hill: "), "Slope factor (steepness of the curve)"]),
                        html.Li([html.B("s: "), "Asymmetry factor (s=1 makes it symmetric like 4PL)"])
                    ], style={'text-align': 'left'})
                ]), 
                className="mb-3", 
                style={"background-color": "rgba(14, 47, 68, 0.3)"}
            )
        ])
    
    elif model_type == 'exp':
        return html.Div([
            dbc.Card(
                dbc.CardBody([
                    html.H4("Exponential Model", className="card-title"),
                    html.P("Formula: y = a * (1 - e^(-b*x)) + c"),
                    html.Hr(),
                    html.Ul([
                        html.Li([html.B("a: "), "Amplitude (height of the curve)"]),
                        html.Li([html.B("b: "), "Rate constant (how quickly response changes with dose)"]),
                        html.Li([html.B("c: "), "Offset (vertical shift of the entire curve)"])
                    ], style={'text-align': 'left'})
                ]), 
                className="mb-3", 
                style={"background-color": "rgba(14, 47, 68, 0.3)"}
            )
        ])
    
    return html.Div()

def default_parameter_explanations(model_type):
    # Similar to parameter_explanations but with default explanations
    return parameter_explanations(model_type)

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
        template='vapor',
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
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"R² = {r2:.4f}, RMSE = {rmse:.4f}",
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
        template='vapor',
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

if __name__ == '__main__':
    app.run_server(debug=True)