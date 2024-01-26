import pathlib
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import Dash, dcc, dash_table, html, Input, Output, callback_context, callback
import dash_bootstrap_components as dbc
import pickle
from Wbal import Wbal3


dash.register_page(__name__,
                   path='/tips',  # represents the url text
                   name='Test test',  # name of page, commonly used as name of link
                   title='Tips'  # epresents the title of browser's tab
)


# df = get_pandas_data("TT_data.pickle")
# Example usage to load a .pickle file
# file_path = f"C://Users//teunv//Dropbox (Personal)//2024//TT analysis//Data23//TT_data.pickle"
# with open(file_path, 'rb') as file:
#     # Load the data from the pickle file
#     df = pd.read_pickle(file)
#

from test import get_pandas_data
df = get_pandas_data("TT_data.pickle")

dfnames = df["TTs"]
dfnames['Date'] = pd.to_datetime(dfnames['Date'], format='%d-%m-%Y')  # Convert to datetime if not already done

# Sort the DataFrame by the "Date" column in descending order
sorted_df = dfnames.sort_values(by='Date', ascending=False)

# Select the first row (which now has the latest date)
latest_row = sorted_df.iloc[0]
latest_name = latest_row['Name']




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
                )


# Layout section: Bootstrap (https://hackerthemes.com/bootstrap-cheatsheet/)
# ************************************************************************
layout = dbc.Container([
   dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='races', multi=False, searchable=True, placeholder="select race",
                         options=[{'label' : x, 'value': x}
                                  for x in sorted(df["TTs"]["Name"].unique())],
                         value=latest_name, persistence=True, persistence_type='session',
                         )
        ],  width={'size': 2},),

       dbc.Col([
           dcc.Dropdown(id='rider', multi=True, searchable=True, placeholder="select riders",
                        persistence=True, persistence_type='memory',
                        # options=[{'label' : x, 'value': x}
                        #           for x in sorted(df_ridersdata["Lastname"].unique())],
                        # [[j for j in range(5)] for i in range(5)]
                        #                          options="",
                        #                          value ='',
                        )
       ], width={'size': 3}, ),
       ]),


dbc.Row([
        dbc.Col([
            dcc.RangeSlider(
                id='slider',
                # marks={i: '{}'.format(i) for i in range(0,int(maxdistance),5)},
                step=0.1,
                min=0,
                value=[0, 2],


            )

        ]
        ),
    ]),

dbc.Row([
        dbc.Col([
            dcc.Graph(id='fig1', figure={})
        ], width={'size': 4, 'offset': 0},

        ),

    dbc.Col([
        dcc.Graph(id='fig2', figure={})
    ], width={'size': 6, 'offset': 0},

    ),

    dbc.Col([
        dash_table.DataTable (
            id="datatable",
            editable=True
                      )
    ], width={'size': 1, 'offset': 0}),


        ]),




dbc.Row([
        dbc.Col([
            dcc.Dropdown(id='para1', multi=False, searchable=True, placeholder="select parameter",
                         value ='power',
                        options=[{'label' : 'power', 'value': 'power'},
                                  {'label' : 'speed', 'value': 'speed'},
                                  {'label' : 'watts/kg', 'value': 'watts/kg'},
                                  {'label' : 'hr', 'value': 'hr'},
                                  {'label' : 'cad', 'value': 'cad'},
                                  {'label' : 'watts/cp', 'value': 'watts/cp'},
                                  {'label' : 'watts/mean_watts', 'value': 'watts/mean_watts'},
                                 ]
                         )
            ],  width={'size': 1},),

        dbc.Col([
            dcc.Input(
                id='moving',
                type='number',
                placeholder="insert",  # A hint to the user of what can be entered in the control
                debounce=True,  # Changes to input are sent to Dash server only on enter or losing focus
                min=0, max=100, step=5,  # Ranges of numeric value. Step refers to increments
                autoComplete='on',
                size="2",
                value=0,  # Number of characters that will be visible inside box
        ),
        ],width={'size': 1},),

       dbc.Col([
           dcc.Dropdown(id='para2', multi=False, searchable=True, placeholder="select parameter",
                     value ='watts/kg',
                        options=[{'label': 'power', 'value': 'power'},
                                 {'label': 'Wbal', 'value': 'Wbal'},
                                 {'label': '%Wbal', 'value': '%Wbal'},
                                 {'label': 'speed', 'value': 'speed'},
                                 {'label': 'watts/kg', 'value': 'watts/kg'},
                                 {'label': 'hr', 'value': 'hr'},
                                 {'label': 'cad', 'value': 'cad'},
                                 {'label': 'watts/cp', 'value': 'watts/cp'},
                                 {'label': 'watts/mean_watts', 'value': 'watts/mean_watts'},
                                 ]
                        )
       ], width={'size': 2, 'offset': 2}, ),







    ]),

dbc.Row([
        dbc.Col([
            dcc.Graph(id='fig3', figure={})
        ], width={'size': 4, 'offset': 0},

        ),

    dbc.Col([
        dcc.Graph(id='fig4', figure={})
    ], width={'size': 6, 'offset': 0},

    ),
    dbc.Col([
        dash_table.DataTable (
            id="datatable2"
              )

            ], width={'size': 2, 'offset': 0}),



]),

dbc.Row([
    dbc.Col([
           dcc.Dropdown(id='segment', multi=False, searchable=True, placeholder="select segment",
                     value ='',
                        options=[               ]
                        )
       ], width={'size': 2, 'offset': 0}, ),


    dbc.Col([
        dash_table.DataTable (
            id="datatable3"
              )

            ], width={'size': 2, 'offset': 0}),

]),

    dcc.Store(id='stored-racedata', data=[], storage_type='memory'),
    dcc.Store(id='stored-racenewdata', data=[], storage_type='memory'),
    dcc.Store(id='stored-ridersdata', data=[], storage_type='memory'),

    ], fluid=True)


@callback(
    [Output('rider', 'options'),
     Output("slider", "marks"),
     Output("slider", "max"),
     Output('stored-racedata', 'data'),
     Output('slider', 'value'),
     Output('segment', 'options'),],
    [Input('races', 'value')])

def build_graph(race):



    racedata = df[race]
    options = [{'label': x, 'value': x}
               for x in sorted(racedata["rider"].unique())]



    maxdistance = racedata["distance"].max()*2
    marks = {i: str(i/2) for i in range(0, int(maxdistance * 2) + 1)}


    options_seg = [{'label': x, 'value': x}
               for x in sorted(racedata["segments"].unique())]


    column_to_drop = 'level_0'

    # Check if the column exists before dropping
    if column_to_drop in racedata.columns:
        racedata.drop(column_to_drop, axis=1, inplace=True)

    else:
        print(f"Column '{column_to_drop}' does not exist in the DataFrame.")

    racedata.reset_index(inplace=True)
    racedata = racedata.to_dict('records')
    slidervalue = [0, maxdistance]


    return (options, marks, maxdistance, racedata, slidervalue, options_seg)



# make datatable 1 and calculation
@callback(

    [Output('datatable', 'data'),
     Output('datatable', 'columns'),
     Output('stored-ridersdata', 'data')],
    [Input('stored-racedata', 'data'),
     Input('races', 'value')])

def build_graph(racedata, race):
    racedata = pd.DataFrame(racedata)
    rider = racedata["rider"].unique()

    df = get_pandas_data("TT_data.pickle")


    riderdata = df["TTdata"][(df["TTdata"]["Source"] == "new") & (df["TTdata"]["Race"] == race)]


    columns_to_drop = ['CRR', 'BikeWeight','Source','Race']
    riderdata = riderdata.drop(columns=columns_to_drop)
    # ridersdata = pd.read_excel("C:/Users/teunv/Dropbox (Personal)/2024/TT analysis/TT_info_Sim.xlsx", 'Riders')

    df_filt_riders = riderdata[riderdata['Rider'].isin(rider)]
    columns = [{"name": i, "id": i} for i in df_filt_riders.columns]
    ridersdatanew = df_filt_riders.to_dict('records')


    return (ridersdatanew, columns, ridersdatanew)



@callback(

    Output('stored-racenewdata', 'data'),
    [Input('datatable', 'data'),
     Input('stored-racedata', 'data')])

def build_graph(ridersdata, racedata):
    ridersdata = pd.DataFrame(ridersdata)
    racenewdata1 = pd.DataFrame(racedata)


    mean_watts_per_rider = racenewdata1.groupby('rider')['power'].mean()
    for rider, mean_watts in mean_watts_per_rider.items():
        racenewdata1.loc[racenewdata1['rider'] == rider, 'watts/mean_watts'] = racenewdata1.loc[racenewdata1['rider'] == rider, 'power'] / mean_watts

    for index, row in ridersdata.iterrows():       # maakt watts/kg or watts/cp etc
        bmass = int(row['BW'])
        cp = int(row['CP'])
        W = int(row["W"])

        name = row['Rider']
        racenewdata1.loc[racenewdata1['rider'] == name, 'watts/kg'] = racenewdata1.loc[racenewdata1['rider'] == name, 'power'] / bmass
        racenewdata1.loc[racenewdata1['rider'] == name, 'watts/cp'] = racenewdata1.loc[racenewdata1['rider'] == name, 'power'] / cp
        test = racenewdata1.loc[racenewdata1['rider'] == name]

        column_to_drop = 'level_0'

        # Check if the column exists before dropping
        if column_to_drop in test.columns:
            test.drop(column_to_drop, axis=1, inplace=True)

        else:
            print(f"Column '{column_to_drop}' does not exist in the DataFrame.")


        test_reset = test.reset_index()



        racenewdata1.loc[racenewdata1['rider'] == name,  'Wbal']  = Wbal3(test_reset, cp, W)
        racenewdata1.loc[racenewdata1['rider'] == name, '%Wbal'] = Wbal3(test_reset, cp, W)/W
        print(racenewdata1[["power","Wbal"]])
        # wbal = Wbal3(test_reset, 400, W)

        # racenewdata.loc[racenewdata['rider'] == name,  'wbal'] = Wbal3(racenewdata.loc[racenewdata['rider'] == name], cp, W)

        # Wbal3(df_data, CP, W)



    racenewdata2 = racenewdata1.to_dict('records')


    return (racenewdata2)
#



@callback(
    [Output('fig1', 'figure'),
     Output('fig2', 'figure'),
     Output('fig3', 'figure'),
     Output("fig4", 'figure'),
     Output('datatable2', 'data'),
     Output('datatable3', 'data'),],
    [Input('slider', 'value'),
     Input('moving', 'value'),
     Input('stored-racenewdata', 'data'),
     Input('rider', 'value'),
     Input('para1', 'value'),
     Input('para2', 'value'),
     Input('segment', 'value')])

def build_graph(slider, moving, racenewdata, riders, para1, para2, segment):

    racedata = pd.DataFrame(racenewdata)
    racedata.set_index('index', inplace=True)
    df_filt_riders = racedata[racedata['rider'].isin(riders)]

    fastest = riders[0]
    # df_results = analyseddata.groupby('rider').mean()
    # moving = 60
    if moving > 0:
        for rider in riders: # loop the riders and parameters on the second y axis
            df_filt_riders.loc[df_filt_riders['rider'] == rider, para1] = df_filt_riders.loc[df_filt_riders[
                                                                                            'rider'] == rider, para1].rolling(window=moving).mean()
            df_filt_riders.loc[df_filt_riders['rider'] == rider, para2] = df_filt_riders.loc[df_filt_riders[
                                                                                                 'rider'] == rider, para2].rolling(
                window=moving).mean()

            #                                                                     racenewdata[
            #                                                                                 'rider'] == name, 'watts'] / bma
            # df_filt_riders[rider] = df_filt_riders[rider].rolling(window=moving).mean()



    # from here data is chosen by sliders
    analyseddata = df_filt_riders[(df_filt_riders["distance"] > slider[0]/2) & (df_filt_riders["distance"] < slider[1]/2)]
    min_secs_per_rider = analyseddata.groupby('rider')['secs'].min()
    # Step 2: Subtract the minimum 'secs' value from each row for the corresponding rider
    analyseddata.loc[:,'secs_difference'] = analyseddata.apply(lambda row: row['secs'] - min_secs_per_rider[row['rider']], axis=1)

    for rider in riders:
        analyseddata.loc[analyseddata['rider'] == rider, 'timediff'] = analyseddata.loc[analyseddata['rider'] == rider, 'secs_difference'] - analyseddata.loc[analyseddata['rider'] == fastest, 'secs_difference']



    fig1 = px.scatter_mapbox(analyseddata, lat="lat", lon="lon", color="rider",
                            zoom=12,
                            mapbox_style="open-street-map")



    fig1.update_layout(showlegend=False)

    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4 = make_subplots(specs=[[{"secondary_y": True}]])

    dfalt = analyseddata[analyseddata["rider"] == riders[0]]

    ymin = dfalt["alt"].min()
    ymax = dfalt["alt"].max()

    colors= dfalt["segments"]



    fig2.add_trace(
        go.Scatter(x=dfalt["distance"], y=dfalt["alt"], stackgroup='one', opacity=1, name="alt",mode='markers+lines',
                   marker=dict(size=6,
                               color=colors,
                               colorscale="Plasma",
                               showscale=True),
                   line=dict(width=1, color='rgb(224, 224, 224)')
                   ),
        secondary_y=False,
    )

    fig3.add_trace(
        go.Scatter(x=dfalt["distance"], y=dfalt["alt"], stackgroup='one', opacity=0, name="alt",
                   line=dict(width=1, color='rgb(224, 224, 224)')
                   ),
        secondary_y=False,
    )

    fig4.add_trace(
        go.Scatter(x=dfalt["distance"], y=dfalt["alt"], stackgroup='one', opacity=0, name="alit",
                   line=dict(width=1, color='rgb(224, 224, 224)')
                   ),
        secondary_y=False,
    )

    for rider in riders:
        dfnew = analyseddata[analyseddata["rider"] == rider]

        fig2.add_trace(
            go.Scatter(x=dfnew["distance"], y=dfnew["timediff"], name=rider),
            secondary_y=True,
        )

        fig3.add_trace(
        go.Scatter(x=dfnew["distance"], y=dfnew[para1], name=rider),
        secondary_y=True,
        )

        fig4.add_trace(
            go.Scatter(x=dfnew["distance"], y=dfnew[para2], name=rider),
            secondary_y=True,
        )

    fig2.update_layout(
        template='simple_white', showlegend=False,
        yaxis_range=[ymin-10, ymax+10]
    )

    fig2.update_yaxes(
        title_text="time losses",
        secondary_y=False)

    fig2.update_yaxes(
        title_text= "Altitude", secondary_y=False)

    fig3.update_layout(
        template='simple_white', showlegend=True,
        yaxis_range=[ymin-10, ymax+10])

    fig3.update_yaxes(
        title_text=para1,

        secondary_y=True)

    fig3.update_yaxes(
        title_text= "Altitude", secondary_y=False)

    fig4.update_layout(
        template='simple_white', showlegend=True,
        yaxis_range=[ymin-10, ymax+10])

    fig4.update_yaxes(
        title_text=para2, secondary_y=True)

    fig4.update_yaxes(
        title_text= "Altitude", secondary_y=False)


    # make table 2
    df_results = analyseddata.groupby('rider').agg({'power': ['max', 'mean'], 'speed': ['mean'],'watts/kg': ['mean'],"timediff": "last"}).round(1)
    df_results.columns = df_results.columns.droplevel()
    df_results.reset_index(inplace=True)
    df_results.columns = ["", "max W","avg W", "avg Speed","avg W/kg",'Timediff']




    #make table with segements
    if len(riders) > 1:
        df_filt_riders = df_filt_riders[df_filt_riders['segments'] == segment]


    df_results_segment = df_filt_riders.groupby(['rider','segments']).agg({
            'distance': ['first',"last"],
            'pacing': ["mean"],
            'watts/cp': ["mean"],
            'power': ['mean'],
            'speed': ['mean'],
            'watts/kg': ['mean'],
            'watts/mean_watts': ["mean"],
            'secs': lambda x: x.iloc[-1] - x.iloc[0],

        }).round(2)

    df_results_segment_reset = df_results_segment.reset_index()
    df_results_segment_reset.columns = ['Name','Segments', 'Seg_start (KM)','Seg_end (KM)', 'pacing (%CP)','W/cp (%)','watts (W)', 'kph (KM/h)', 'watts/kg (W/kg)','W/avgp (%)','time (S)']
    df_results_sorted_desc = df_results_segment_reset.sort_values(by='Segments', ascending=True)

    datatable3 = df_results_sorted_desc.to_dict('records')
    datatable2 = df_results.to_dict('records')
    return (fig1, fig2, fig3, fig4, datatable2, datatable3)




if __name__ == '__main__':
    app.run_server(debug=False)