# index.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
from app import app
from pages import page1, page2

app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),

    dbc.NavbarSimple(
        brand="Doser Responser",
        brand_href="/",
        color="primary",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="/", id="nav-home")),
            dbc.NavItem(dbc.NavLink("File Uploader", href="/page1", id="nav-page1")),
            dbc.NavItem(dbc.NavLink("Visualizer", href="/page2", id="nav-page2")),
        ],
    ),

    html.Div(id='page-content'),
    dcc.Store(id='shared-data', storage_type='local'),
], fluid=True)

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return html.H3('Welcome to the home page')
    elif pathname == '/page1':
        return page1.layout
    elif pathname == '/page2':
        return page2.layout
    else:
        return '404 - Page not found'

@app.callback(
    [Output('nav-home', 'active'),
     Output('nav-page1', 'active'),
     Output('nav-page2', 'active')],
    Input('url', 'pathname')
)
def update_active_navlink(pathname):
    return [
        pathname == '/',
        pathname == '/page1',
        pathname == '/page2'
    ]

if __name__ == '__main__':
    app.run_server(debug=True)