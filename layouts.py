import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


head_div_style = {'padding': '0.25%', 'text-align': 'center'}
big_div_style = {'padding': '0.25%', 'padding-top': '0.5%', 'padding-bottom': '0.5%'}


layout = html.Div([
    dcc.Location(id='url', refresh=True),
    dcc.Interval(id='interval2',
                 interval=40,  # era 80 para los modelos 1 al 3. 40 para el modelo 4 final
                 n_intervals=0),
    html.Div([  # header
        html.Div([
            html.Img(
                src='/assets/Tedep.png',
                height='70 px',
                width='auto')
        ], className='col-2', style=head_div_style),
        html.Div([
            html.H1('Proyecto final', id='titulo', style={'font-size': 70})
        ], className='col-8'),
        html.Div([
            html.Img(
                src='/assets/UTN_logo.png',
                height='70 px',
                width='auto')
        ], className='col-2', style=head_div_style)
    ], className='row align-items-center', style=head_div_style),  # header
    html.Hr(),
    html.Br(),

    html.Div([
        dbc.Row([dcc.Checklist(id='modo_admin', options=[{"label": "Modo Admin", "value": "Modo Admin"}], value=[])]),
        html.Br(),
        dbc.Row([
            html.Div([html.H1("CLASIFICACIÓN", style={'font-size': 37})
                      ], id='texto-principal', className='my-5', style={'text-align': 'center', 'display': 'block'}),
            html.Div([
                dcc.RadioItems(id='checklist',
                               options=[{'label': 'Seleccione una opción para guardar datos', 'value': 'nada'},
                                        {'label': '\t\t\tLeer datos de movimientos cuadrados', 'value': 'cuadrado'},
                                        {'label': '\t\t\tLeer datos de movimientos circulares', 'value': 'circular'},
                                        {'label': '\t\t\tLeer datos de módulo en reposo', 'value': 'reposo'}],
                               value='nada', labelStyle={'display': 'block', 'margin-left': '5%', 'margin-up': '5%'})
                         ], id='checklist-div', style={'display': 'none'}),
        ])
    ], className='align-items-center', style=big_div_style)
], className='container')
