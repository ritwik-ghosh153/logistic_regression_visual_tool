import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import dash_table
import plotly.io as pio
import sklearn.datasets as d
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
server = app.server

# @server.route("/dash")
# def MyDashApp():
#     app.title = "Covid-19 Insights"
#     return app.index()


# setting initial values


data = pd.DataFrame()
n_classes = 2
# n_clusters = 1
n_samples = 300

isDataSet = False
isDataGenerated = False
canTrain = False

multi_class_options = [
    {
        'label': 'auto', 'value': 'auto',
    },
    {
        'label': 'ovr', 'value': 'ovr',
    },
    {
        'label': 'multinomial', 'value': 'multinomial',
    },
]

penalty_options = [
    {
        'label': 'l1', 'value': 'l1',
    },
    {
        'label': 'l2', 'value': 'l2',
    },
    {
        'label': 'elasticnet', 'value': 'elasticnet',
    },
    {
        'label': 'none', 'value': 'none’',
    },
]

solver_options = [
    {
        'label': 'newton-cg', 'value': 'newton-cg',
    },
    {
        'label': 'lbfgs', 'value': 'lbfgs',
    },
    {
        'label': 'liblinear', 'value': 'liblinear',
    },
    {
        'label': 'sag', 'value': 'sag',
    },
    {
        'label': 'saga', 'value': 'saga',
    },
]

app.layout = html.Div(children=
[
    html.H1('Logistic Regression Visual Tool'),
    html.Hr([]),
    # main playground
    html.Div(
        [

            # dataset generation
            html.Div([
                html.Div([
                    html.H3('Dataset generation'),
                ], className='col-md-12'),

                html.Div([
                    # classes parameter
                    html.Div([
                        html.P("Number of classes in data", className="hyperparameter-title"),
                        dcc.Slider(
                            id='classes_picker',
                            min=2,
                            max=4,
                            step=1,
                            value=2,
                        ),
                    ], className='slider'
                    ),

                    # clusters parameter
                    # html.Div([
                    #     html.P("Number of clusters in data", className="hyperparameter-title"),
                    #     dcc.Slider(
                    #         id='clusters_picker',
                    #         min=1,
                    #         max=5,
                    #         step=1,
                    #         value=1,
                    #     ),
                    # ], className='slider'
                    # ),

                    # samples parameter
                    html.Div([
                        html.P("Number of samples in data", className="hyperparameter-title"),
                        dcc.Slider(
                            id='samples_picker',
                            min=10,
                            max=1000,
                            step=5,
                            value=300,
                        ),
                    ], className='slider'
                    ),

                    # data generation button
                    html.Button(
                        id='data-generator',
                        children=['Generate'],
                    ),
                ], className='col-md-6'),
                # data graph
                html.Div([
                    html.Div([
                        dcc.Graph(
                            id='dataset_graph',
                            # figure='fig'
                        ),
                    ]),

                ], className='col-md-6'),

            ], className='row'),
            # upper half
            html.Div([
                # tuners
                html.Div([
                    html.Button(
                        id='model-trainer',
                        children=['Train'],
                    ),
                    html.H1('Tuners here'),

                    # penalty
                    html.Div([
                        html.P("Penalty parameter", className="hyperparameter-title"),
                        dcc.Dropdown(
                            id='penalty_picker',
                            options=penalty_options,
                            value='l2'
                        ),
                    ], className='dropdown'
                    ),

                    # dual
                    html.Div([
                        html.P("Dual parameter", className="hyperparameter-title"),
                        dcc.RadioItems(
                            id='dual_picker',
                            options=[
                                {'label': 'True', 'value': 'True'},
                                {'label': 'False', 'value': 'False'},
                            ],
                            value='False'
                        ),
                    ], className="radio", id='dual-div') if True else html.H1('Dual here'),

                    # C parameter
                    html.Div([
                        html.P("C parameter", className="hyperparameter-title"),
                        dcc.Slider(
                            id='c_picker',
                            min=0,
                            max=5,
                            step=0.1,
                            value=1.0,
                        ),
                    ], className='slider'
                    ),

                    # fit_intercept
                    html.Div([
                        html.P("Fit intercept", className="hyperparameter-title"),
                        dcc.RadioItems(
                            id='fit_intercept_picker',
                            options=[
                                {'label': 'True', 'value': 'True'},
                                {'label': 'False', 'value': 'False'},
                            ],
                            value='True'
                        ),
                    ], className="radio"),

                    # random_state parameter
                    html.Div([
                        html.P("random_state", className="hyperparameter-title"),
                        dcc.Slider(
                            id='random_state_picker',
                            min=0,
                            max=10,
                            step=1,
                            value=0,
                        ),
                    ], className='slider'
                    ) if True else html.H1('random state picker here'),

                    # solver
                    html.Div([
                        html.P("Solver parameter", className="hyperparameter-title"),
                        dcc.Dropdown(
                            id='solver_picker',
                            options=solver_options,
                            value='lbfgs'
                        ),
                    ], className='dropdown'
                    ),

                    # max_iter
                    html.Div([
                        html.P("Max Iter parameter", className="hyperparameter-title"),
                        dcc.Slider(
                            id='max_iter_picker',
                            min=1,
                            max=300,
                            step=1,
                            value=100,
                        ),
                    ], className='slider'
                    ),

                    # multi_class
                    html.Div([
                        html.P("Solver parameter", className="hyperparameter-title"),
                        dcc.Dropdown(
                            id='multi_class_picker',
                            options=multi_class_options,
                            value='lbfgs'
                        ),
                    ], className='dropdown'
                    ),

                ],
                    className='col-md-6'),
                # graph
                html.Div([
                    html.H1('Graph here'),
                    html.Div([
                        dcc.Graph(
                            id='trained_model_graph',
                        ),
                    ]),
                ],
                    className='col-md-6'),
            ], className='row'),
            # lower half
            html.Div([
                html.Div(
                    [
                        html.H1('Data info here'),
                    ], className='col-md-6'
                ),
                html.Div(
                    [
                        html.H1('Results here'),
                    ], className='col-md-6'
                ),
            ], className='row'),
        ], id="playground"),
    html.Div([], id='asd')
],
    className='container')


# functions

@app.callback(Output('asd', 'style'),
              [Input('classes_picker', 'value'), Input('samples_picker', 'value')])
def set_data(classes, samples):
    print('set data')
    # global n_clusters
    global n_classes
    global n_samples
    global isDataSet
    n_classes = classes
    # n_clusters = clusters
    n_samples = samples

    # isDataSet = True
    print('set data 2')
    return {}


@app.callback(Output('dataset_graph', 'figure'), [Input('data-generator', 'n_clicks')])
def generate_data(clicks):
    # if not clicks:
    #     return
    # global n_clusters
    global n_classes
    global isDataSet
    global n_samples
    global data
    global isDataGenerated

    if not isDataGenerated:
        isDataGenerated = True
        return go.Figure()

    # isDataGenerated = False
    # if not isDataSet:
    #     n_classes = 2
    #     n_samples = 300
    #     isDataSet = True

    isDataGenerated = False

    print('generate data')
    x = d.make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_repeated=0, n_redundant=0,
                              n_classes=n_classes, n_clusters_per_class=1, shift=None)
    data = pd.DataFrame(x[0]).join(pd.DataFrame(x[1], columns=['Labels']))
    # print(data)

    traces = []
    for i in range(0, n_classes):
        x = data[data['Labels'] == i]
        traces.append(go.Scatter(x=x[0], y=x[1], mode='markers', name='class ' + str(i)))

    # train_model(0)
    isDataGenerated = True
    print('generate data 2')
    return go.Figure(data=traces)


@app.callback(Output('trained_model_graph', 'figure'), [Input('model-trainer', 'n_clicks')])
def train_model(clicks):
    # if clicks == 0:
    #     return
    # global n_clusters
    global n_classes
    global n_samples
    global data
    global isDataGenerated
    global canTrain

    if not canTrain or not isDataGenerated:
        canTrain = True
        return go.Figure()

    # if not isDataGenerated:
    #     return

    print('training model')
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:2], data.iloc[:, 2], test_size=0.2)

    model = LogisticRegression()
    trained_model = model.fit(x_train, y_train)
    prediction_train = trained_model.predict(x_train)
    prediction_test = trained_model.predict(x_test)
    prediction_total = trained_model.predict(data.iloc[:, 0:2])
    coef = trained_model.coef_
    # print(trained_model.intercept_)
    # y = coef[0][0] * data[0] + coef[0][1]
    y = -(data[0] * coef[0][0] + trained_model.intercept_[0]) / coef[0][1]

    traces = []
    for i in range(0, n_classes):
        x = data[data['Labels'] == i]
        traces.append(go.Scatter(x=x[0], y=x[1], mode='markers', name='class ' + str(i)))

    traces.append(go.Scatter(x=data[0], y=y, mode='lines'))

    print('training model 2')
    return go.Figure(data=traces)


if __name__ == "__main__":
    app.run_server(debug=True)
