import dash
from dash_bootstrap_templates import load_figure_template


app = dash.Dash(__name__, serve_locally=True, suppress_callback_exceptions=True)
load_figure_template("minty")
server = app.server
app.title = 'Rowing Corrector'
