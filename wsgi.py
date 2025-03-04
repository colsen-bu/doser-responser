# This file serves as the entry point for production servers

# Import the Flask server object
from dash_app.app import server

# Import the layout from index.py to ensure it's loaded
import dash_app.index

# This allows gunicorn to find the server object
if __name__ == "__main__":
    server.run(debug=False, host='0.0.0.0')