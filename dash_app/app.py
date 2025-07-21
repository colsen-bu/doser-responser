from dash import Dash
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR], suppress_callback_exceptions=True)
server = app.server

# Import index to register layout and callbacks when module is loaded
import index

# Only run the dev server when this file is executed directly
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')