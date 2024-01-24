import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server

sidebar = dbc.Nav(
            [
                dbc.NavLink(
                    [
                        html.Div(page["name"], className="ms-2"),
                    ],
                    href=page["path"],
                    active="exact",

                )
                for page in dash.page_registry.values()
            ],
            vertical=True,
            pills=True,
            className="bg-light",
)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("Tudor TT analysis",
                         style={'fontSize':50, 'textAlign':'center'}))
    ]),

    html.Hr(),

    dbc.Row(
        [
            dbc.Col(
                [
                    sidebar
                ], xs=2, sm=2, md=2, lg=1, xl=1, xxl=1),

            dbc.Col(
                [
                    dash.page_container
                ], xs=10, sm=10, md=11, lg=11, xl=11, xxl=11)
        ]
    ),

    dcc.Store(id="store", data={}),

], fluid=True)


if __name__ == "__main__":
    app.run(debug=True)
