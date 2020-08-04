import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input,Output
import dash_table
import plotly.io as pio


# external CSS stylesheets
external_stylesheets = [
   {
       'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
       'rel': 'stylesheet',
       'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
       'crossorigin': 'anonymous'
   }
]

pio.templates.default = "plotly_dark"

app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                # meta_tags=[
                #     {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                # ]
                )
server=app.server

# @server.route("/dash")
# def MyDashApp():
#     app.title = "Covid-19 Insights"
#     return app.index()


app.layout = html.Div(children=
                      [
                          html.H1('Logistic Regression Visual Tool',
                                  style= {'color':'white', 'text-align':'center'}
                                  ),
                      ],
    className='col')

if __name__ == "__main__":
    app.run_server(debug=True)