import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
import pandas as pd
from dash import Dash, dcc, dash_table, html, Input, Output, callback_context, callback, ctx
import dash_bootstrap_components as dbc
import pickle
from datetime import datetime
from dash_iconify import DashIconify
icon = DashIconify(icon="carbon:save")
# from cheungsmethod import n_wind_to_degrees
from cheungsmethod import cheungsmethod
from Wbal import Wbal3
from test import get_pandas_data
from test import save_pandas_data
import ast
drivechain = 0.97
# crr = 0.003
# body_mass = 70
# system_mass = 8
# total_mass = body_mass+system_mass
gravity = 9.81
clicks = 0

# density = 1.233


# To create meta tag for each page, define the title, image, and description.
dash.register_page(__name__,
                   path='/',  # '/' is home page and it represents the url
                   name='Replay',  # name of page, commonly used as name of link
                   title='Replay', # title that appears on browser's tab

                   # image='pg1.png',  # image in the assets folder
                   # description='Histograms are the new bar charts.'
)

# Laad data
# def get_pandas_data(pickle_filename: str) -> pd.DataFrame:
#     '''
#     Load data from /data directory as a pandas DataFrame
#     using relative paths. Relative paths are necessary for
#     data loading to work in Heroku.
#     '''
#     PATH = pathlib.Path(__file__).parent
#     DATA_PATH = PATH.joinpath("data").resolve()
#     # Assuming the file is a pickled DataFrame
#     with open(DATA_PATH.joinpath(pickle_filename), 'rb') as file:
#         data = pickle.load(file)
#     return data

df = get_pandas_data("TT_data.pickle")
# Example usage to load a .pickle file
# file_path = f"C://Users//teunv//Dropbox (Personal)//2024//TT analysis//Data23//TT_data.pickle"
# with open(file_path, 'rb') as file:
#     # Load the data from the pickle file
#     df = pd.read_pickle(file)

dfnames = df["TTs"]
dfnames['Date'] = pd.to_datetime(dfnames['Date'], format='%d-%m-%Y')  # Convert to datetime if not already done

# Sort the DataFrame by the "Date" column in descending order
sorted_df = dfnames.sort_values(by='Date', ascending=False)

# Select the first row (which now has the latest date)
latest_row = sorted_df.iloc[0]
latest_name = latest_row['Name']


layout = dbc.Container([
   dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='TTs', multi=False, searchable=True, placeholder="select race",
                         options=[{'label': x, 'value': x}
                                    for x in sorted(df["TTs"]["Name"].unique())],
                         value =latest_name, persistence=True, persistence_type='session',
                         )
        ],  width={'size': 2},),

       dbc.Col([
           dcc.Dropdown(id='Rider', multi=False, searchable=True, placeholder="select rider",
                        value='Suter', persistence=True, persistence_type='session',
                        )
       ], width={'size': 2}, ),

]),


    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line-fig', figure={}),
                ],  width={'size': 8},),

        dbc.Col([
            dcc.Graph(id='line-fig2', figure={}),
        ], width={'size': 4, 'offset':0}, ),
]),


    dbc.Row([
        dbc.Col([
            dbc.Label("Set CdA"),
            dcc.Slider(
                id='sl_CdA',
                step=0.001,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ],  width={'size': 3},
        ),

        # dbc.Col([
        #     dbc.Checkbox(
        #         id= "checkbox",
        #         label="set",
        #     )
        # ],  width={'size': 1},
        # ),

        dbc.Col([
            dbc.Label("Set Bodyweight"),
            dcc.Slider(
                id='sl_BW',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
                step=0.5,
                # min=0.15,
                # max=0.25,
                # value=CdA,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ],  width={'size': 3},
        ),

        dbc.Col([
            dbc.Label("Set CP"),
            dcc.Slider(
                id='sl_CP',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
                step=10,
                # min=0.15,
                # max=0.25,
                # value=CdA,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ], width={'size': 3, "offset": 2},
        ),

]),

dbc.Row([
        dbc.Col([
            dbc.Label("Set CRR"),
            dcc.Slider(
                id='sl_CRR',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
                step=0.001,
                # min=0.15,
                # max=0.25,
                # value=CdA,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ],  width={'size': 3},
        ),

        # dbc.Col([
        #     dbc.Checkbox(
        #         id= "checkbox1",
        #         label="set",
        #     )
        # ],  width={'size': 1},
        # ),

        dbc.Col([
            dbc.Label("Set Density"),
            dcc.Slider(
                id='sl_Density',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
                step=0.001,
                # min=0.15,
                # max=0.25,
                # value=CdA,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ],  width={'size': 3},
        ),

    dbc.Col([
        dbc.Label("Set w"),
        dcc.Slider(
            id='sl_W',
            # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
            # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
            step=100,
            # min=0.15,
            # max=0.25,
            # value=CdA,
            tooltip={"placement": "bottom", "always_visible": True},
            persistence=True, persistence_type='session',
        )
    ], width={'size': 3, "offset": 2},
    ),

]),



dbc.Row([
        dbc.Col([
            dbc.Label("Set Wind-Direction"),
            dcc.Slider(
                id='sl_WindDirection',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                marks={
                    0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: "SW", 270: 'W', 315: 'NW'},
                step= 22.5,
                min=0,
                max=360,
                value=337.5,
                tooltip={"placement": "bottom", "always_visible": True},
                persistence=True, persistence_type='session',
            )
        ],  width={'size': 3},
        ),

        # dbc.Col([
        #     dbc.Checkbox(
        #         id= "checkbox2",
        #         label="Set wind-Strength",
        #     )
        # ],  width={'size': 1},
        # ),

        dbc.Col([
            dbc.Label("Set Wind strenght"),
            dcc.Slider(
                id='sl_Windstrenght',
                # marks={i: str(i) for i in np.arange(0.15, round(0.25), 0.01)},
                # marks={round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA-0.05, CdA+0.05, 0.01)},
                step=1,
                min=0,
                max=40,
                value=10,
                persistence=True, persistence_type='session',

                tooltip={"placement": "bottom", "always_visible": True},
            )
        ],  width={'size': 3},
        ),
]),

dbc.Row([
    dbc.Col(html.Div("", style={'height': '100px', 'textAlign': 'center'}))
]),

dbc.Row([
    dbc.Col([
        dash_table.DataTable (
            id="datatable4"
              )

            ], width={'size': 4, 'offset': 1}),



    dbc.Col([
        dbc.Button(icon, id="button", size="lg", color="success", n_clicks=0),
        ], width={'size': 4, 'offset': 1})
    ]),

    ], fluid=True)

@callback(
    Output('Rider','options'),
    [Input('TTs', 'value')])

def build_graph(TT):

    df_1 = df[TT]
    options = [{'label': x, 'value': x}
               for x in sorted(df_1["rider"].unique())]


    return options

@callback(
    [Output('sl_CdA','value'),
     Output('sl_CdA','min'),
     Output('sl_CdA','max'),
     Output('sl_CdA','marks'),
     Output('sl_BW', 'value'),
     Output('sl_BW', 'min'),
     Output('sl_BW', 'max'),
     Output('sl_BW', 'marks'),
     Output('sl_CRR', 'value'),
     Output('sl_CRR', 'min'),
     Output('sl_CRR', 'max'),
     Output('sl_CRR', 'marks'),
     Output('sl_CP', 'value'),
     Output('sl_CP', 'min'),
     Output('sl_CP', 'max'),
     Output('sl_CP', 'marks'),
     Output('sl_W', 'value'),
     Output('sl_W', 'min'),
     Output('sl_W', 'max'),
     Output('sl_W', 'marks'),
     Output('sl_Windstrenght', 'value'),
     Output('sl_Windstrenght', 'min'),
     Output('sl_Windstrenght', 'max'),
     Output('sl_Windstrenght', 'marks'),
     Output('sl_WindDirection', 'value'),
     Output('sl_Density', 'value'),
     Output('sl_Density', 'min'),
     Output('sl_Density', 'max'),
     Output('sl_Density', 'marks'),


     ],
    [Input('TTs', 'value'),
     Input('Rider', 'value')])

def build_graph(tt, Rider):


    TTdata = df["TTdata"][(df["TTdata"]["Race"] == tt) & (df["TTdata"]["Source"] == "new")]
    riderdata = TTdata[TTdata["Rider"] == Rider]
    riderdata .reset_index(drop=True, inplace=True)

    # Set BW
    BW = round(riderdata["BW"][0], 1)
    min_BW  = BW-6
    max_BW  = BW+6
    # marks_BW = {round(i, 1): str('%.1f' % round(i, 1)) for i in np.arange(BW - 5, BW + 5, 3.0)}
    marks_BW = {i: str(i) for i in range(min_BW, max_BW, 2)}

    #set CdA
    CdA = round(riderdata["CdA"][0], 3)
    marks_cda = {round(i, 2): str('%.2f' % round(i, 2)) for i in np.arange(CdA - 0.05, CdA + 0.05, 0.02)}
    min_cda = CdA-0.05
    max_cda = CdA+0.05

    # set CRR
    CRR= round(riderdata["CRR"][0], 4)
    min_CRR = 0
    max_CRR = CRR+0.005
    marks_CRR = {round(i, 4): str('%.4f' % round(i, 4)) for i in np.arange(0, max_CRR, 0.002)}

    # marks_CRR = {i: str(i) for i in range(min_CRR, max_CRR, 0.001)}

    #Set Density
    Density = round(riderdata["Density"][0], 4)
    min_Density  = Density  - 0.05
    max_Density  = Density  + 0.05
    marks_Density = {round(i, 3): str('%.3f' % round(i, 3)) for i in np.arange(min_Density, max_Density, 0.02)}

    #set Winddirection
    Winddirection = round(riderdata["Wind"][0], 1)

    #Set Windspeed
    Windspeed = round(riderdata["Windspeed"][0], 1)
    min_Windspeed = Windspeed - 15
    max_Windspeed = Windspeed + 15
    marks_Windspeed = {i: str(i) for i in range(min_Windspeed, max_Windspeed, 2)}

    # Set CP
    CP = round(riderdata["CP"][0], 1)
    min_CP  = CP-25
    max_CP  = CP+25
    marks_CP = {i: str(i) for i in range(min_CP, max_CP, 5)}

    # Set W
    W = round(riderdata["W"][0], 1)
    min_W  = W-2500
    max_W  = W+2500
    marks_W = {i: str(i) for i in range(min_W, max_W, 500)}


    return CdA, min_cda, max_cda, marks_cda, BW, min_BW, max_BW, marks_BW, CRR, min_CRR, \
           max_CRR, marks_CRR, CP, min_CP, max_CP, marks_CP, W, min_W, max_W, marks_W, \
           Windspeed, min_Windspeed, max_Windspeed, marks_Windspeed, Winddirection, \
           Density, min_Density, max_Density, marks_Density




@callback(
    [Output('line-fig','figure'),
    Output('line-fig2','figure'),
    Output('datatable4', 'data')],
    [Input('Rider', 'value'),
     Input('TTs', 'value'),
     Input('sl_CdA', 'value'),
     Input('sl_BW', 'value'),
     Input('sl_WindDirection', 'value'),
     Input('sl_Windstrenght', 'value'),
     Input('sl_CRR', 'value'),
     Input('sl_CP', 'value'),
     Input('sl_W', 'value'),
     Input('sl_Density', 'value'),
     Input('button','n_clicks')])



def build_graph(rider, TT, sl_CdA, BW, direction, strenght, CRR, CP, W, Density, clicks):


    df_data = df[TT][df[TT]["rider"] == rider]  # get data
    df_data.loc[:,'speed']=df_data['speed']/3.6

    CdA = sl_CdA
    CP = CP
    W = W
    density= Density
    TTdata = df["TTdata"][df["TTdata"]["Race"] == TT]

    riderdata = TTdata[TTdata["Rider"] == rider]
    riderdata.reset_index(drop=True, inplace=True)

    # Make datatable
    source_condition1 = riderdata['Source'] == "new"
    c = riderdata.loc[source_condition1].index
    int_value_c = int(c.values[0])


    riderdata.loc[int_value_c, ['Race','Rider','BikeWeight']] = riderdata.loc[0, ['Race','Rider','BikeWeight']]
    riderdata.loc[int_value_c, ['Date']] = datetime.now().strftime('%d-%m-%Y')
    riderdata.loc[int_value_c, ['CP']] = CP
    riderdata.loc[int_value_c, ['W']] = W
    riderdata.loc[int_value_c, ['CdA']] = CdA
    riderdata.loc[int_value_c, ['CRR']] = CRR
    riderdata.loc[int_value_c, ['BW']] = BW
    riderdata.loc[int_value_c, ['Wind']] = direction
    riderdata.loc[int_value_c, ['Windspeed']] = strenght
    riderdata.loc[int_value_c, ['Density']] = density


    BikeW = round(riderdata["BikeWeight"][0], 2)
    total_mass = BikeW+BW

    wind_speed = strenght
    wind_direction = direction
    # wind = add_wind

    # Convert the angle from degrees to radians using NumPy
    angle_rad = np.radians(df_data['bearing_deg'] - wind_direction)

    # Calculate the wind speed in the desired direction
    df_data.loc[:,('wind')] = wind_speed * np.cos(angle_rad)


    df_data.loc[:,'altitude_new'] = cheungsmethod(df_data, drivechain, total_mass, gravity, CRR, density, CdA)

    # cheungs method
    # df_data.loc[:,'altitude_new'] = alt_diff_result
    df_data.loc[:,'altitude_new'] = df_data['altitude_new'].cumsum()

    # add wbal
    df_data.loc[:,'wbal'] = Wbal3(df_data, CP, W)

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    # figure = px.line(df_data, x="distance", y=["alt","altitude_new"], title="Load build up")

    figure.add_trace(
        go.Scatter(x=df_data["distance"], y=df_data["alt"]),
        secondary_y=True,
    )
    figure.add_trace(
        go.Scatter(x=df_data["distance"], y=df_data["altitude_new"]),
        secondary_y=True,
    )

    figure2 = make_subplots(specs=[[{"secondary_y": True}]])
    # figure2 = px.line(df_data, x="distance", y=["wbal", "alt"], title="Wbal")
    figure2.add_trace(
        go.Scatter(x=df_data["distance"], y=df_data["alt"], stackgroup='one', opacity=0, name="alt",
                   line=dict(width=1, color='rgb(224, 224, 224)')
                   ),
        secondary_y=False,
    )

    figure2.add_trace(
        go.Scatter(x=df_data["distance"], y=df_data["wbal"]),
        secondary_y=True,
    )

    figure.update_layout(
        template='simple_white', showlegend=False,
        # yaxis_range=[ymin-10, ymax+10]
    )

    figure2.update_layout(
        template='simple_white', showlegend=False,
        # yaxis_range=[ymin-10, ymax+10]
    )
    # # figure.update_layout(template='simple_white', xaxis={'title': "days"}, yaxis={"title": "altitude"}, hovermode="x")
    #

    datatable4 = riderdata.to_dict('records')
    # save new data when button is used
    if ctx.triggered_id and clicks>0:
        race_condition = df["TTdata"]['Race'] == TT
        rider_condition = df["TTdata"]['Rider'] == rider
        source_condition = df["TTdata"]['Source'] == "new"
        source_condition1 = riderdata['Source'] == "new"

        b = df["TTdata"].loc[race_condition & rider_condition & source_condition].index
        a = riderdata.loc[source_condition1].index

        int_value_a = int(a.values[0])
        int_value_b = int(b.values[0])

        df["TTdata"].loc[int_value_b] = riderdata.loc[int_value_a]
        # save_pandas_data(df, "TT_data.pickle")
        # file_path = r'C:\Users\teunv\Dropbox (Personal)\2024\TT analysis\Data23\TT_data.pickle'
        # # Open the file in binary write mode (wb)
        # with open(file_path, 'wb') as file:
        #     # Dump the data into the file
        #     pickle.dump(df, file)

    return figure, figure2, datatable4