from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR], suppress_callback_exceptions=True)
server = app.server

# Only run the dev server when this file is executed directly
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')