import numpy as np
import pandas as pd
import plotly.graph_objects as goa
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table
import plotly.io as pio
import sklearn.datasets as d
import pandas as pd


# functions

@app.callback(Output('dataset_graph', 'figure'),
              [Input('data-generator'), Input('classes_picker', 'value'), Input('clusters_picker', 'value'),
               Input('samples_picker', 'value'), ])
def generate_data(n_classes, n_clusters, n_samples):
    x = d.make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
                              n_classes=n_classes, n_clusters=n_clusters)
