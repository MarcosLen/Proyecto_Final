from dash.dependencies import Input, Output
from app import app
import callbacks
from layouts import layout


app.layout = layout


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    return layout


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
