from dash import Dash
import dash_bootstrap_components as dbc
import sys
import os


app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
server = app.server

# Only run the dev server when this file is executed directly
if __name__ == '__main__':
    # Add the dash_app directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    # If running from root directory, also add the parent directory
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # When running this file directly, we need to make sure the app object
    # is available as 'app' module for index.py to import
    sys.modules['app'] = sys.modules[__name__]
    
    # Import index to register layout and callbacks BEFORE running the server
    import index
    app.run(debug=True, host='0.0.0.0')