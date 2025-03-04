# python
from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR], suppress_callback_exceptions=True)
server = app.server

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0')