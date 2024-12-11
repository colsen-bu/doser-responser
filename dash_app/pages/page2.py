import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.graph_objs as go
from dash_bootstrap_templates import load_figure_template

# Load the 'vapor' theme
load_figure_template('vapor')

layout = dbc.Container([
    html.H1('Dose Response Curves'),
    html.Div(id='dose-response-graphs'),
], fluid=True)

@callback(
    Output('dose-response-graphs', 'children'),
    Input('shared-data', 'data'),
)
def update_graph(data):
    if data is not None:
        df = pd.DataFrame(data)
        required_columns = {'Dose_uM', 'Response_Metric', 'Treatment'}
        if not required_columns.issubset(df.columns):
            return html.Div('Required columns are missing in the data.')

        # Group data by Treatment and Dose_uM to calculate mean and standard deviation
        grouped = df.groupby(['Treatment', 'Dose_uM']).agg(
            mean_response=('Response_Metric', 'mean'),
            std_response=('Response_Metric', 'std')
        ).reset_index()

        # Create a graph for each Treatment
        graphs = []
        treatments = grouped['Treatment'].unique()
        for treatment in treatments:
            treatment_data = grouped[grouped['Treatment'] == treatment]

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=treatment_data['Dose_uM'],
                y=treatment_data['mean_response'],
                error_y=dict(type='data', array=treatment_data['std_response'], visible=True),
                mode='markers+lines',
                name=treatment,
            ))

            fig.update_layout(
                title=f'Dose Response Curve for {treatment}',
                xaxis_title='Dose (μM)',
                yaxis_title='Response Metric',
                xaxis_type='log',
                template='vapor'
            )

            graphs.append(dcc.Graph(figure=fig))

        return graphs
    else:
        return html.Div('No data available.')