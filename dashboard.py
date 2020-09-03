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
from sklearn.metrics import classification_report, confusion_matrix

import tour

external_scripts = [
    {
        'src': 'https://use.fontawesome.com/releases/v5.0.13/js/solid.js',
        'integrity': 'sha384-tzzSw1/Vo+0N5UhStP3bvwWPq+uvzCMfrN1fEFe+xBmv1C/AtVX5K0uZtmcHitFZ',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://use.fontawesome.com/releases/v5.0.13/js/fontawesome.js',
        'integrity': 'sha384-6OIrr52G08NpOFSZdxxz1xdNSndlD4vdcf/q2myIUVO0VsqaGHJsB0RaBE01VTOY',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://code.jquery.com/jquery-3.3.1.slim.min.js',
        'integrity': 'sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.0/umd/popper.min.js',
        'integrity': 'sha384-cs/chFZiN24E4KMATLdqdvsezGxaGsi4hLGOzlXwp5UZB1LY//20VyM2taTB4QvJ',
        'crossorigin': 'anonymous'
    },
    {
        'src': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.0/js/bootstrap.min.js',
        'integrity': 'sha384-uefMccjFJAIv6A+rW+L4AHf99KvxDjWSu1z9VI8SKNVmz4sk7buKt/6v9KI65qnm',
        'crossorigin': 'anonymous'
    }
]

# external CSS stylesheets
external_stylesheets = [
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

# pio.templates.default = "plotly_dark"

app = dash.Dash(__name__, external_scripts=external_scripts, external_stylesheets=external_stylesheets
                # meta_tags=[
                #     {"name": "viewport", "content": "width=device-width, initial-scale=1"}
                # ]
                )
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <meta name="description" content="A visual tool to visualize and understand change in logistic regression on changing hyper-parameters.">
        <meta name="keywords" content="Logistic Regression, Visualizsation, Logistic, Regression, Data Science, Machine Learning, SKLearn">
        <title>Logistic Regression Visual Tool</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# setting initial values


data = pd.DataFrame()
n_classes = 2
# n_clusters = 1
n_samples = 300

isDataSet = False
isDataGenerated = False
firstTraining = False
firstGeneration = False

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
        'label': 'none', 'value': 'none',
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

penalty_value = 'l2'
dual_value = 'False'
c_value = 1.0
fit_intercept_value = 'True'
random_state_value = 0
solver_value = 'lbfgs'
max_iter_value = 100,
multi_class_value = 'auto'

training_size = 80

dual_show = False
random_state_show = False

dynamic_id_dual = 'dual_picker'
dynamic_id_penalty_picker = 'penalty_picker'
dynamic_id_random_state_picker = 'random_state_picker'

layout_main = html.Div([

    # NAV BAR
    html.Nav([
        html.A("Playground", className="flex-sm-fill text-sm-center nav-link bg-info text-white", href="/"),
        html.A("Tour", className="flex-sm-fill text-sm-center nav-link text-white", href="/tour")
    ], className="nav nav-pills flex-column flex-sm-row bg-dark"),

    html.Div(children=
    [

        html.H1('Logistic Regression Visual Tool', className="text-center"),
        html.Hr([]),
        # main playground
        html.Div(
            [
                # dataset generation
                html.Div([
                    html.Div([
                        html.H4('Dataset generation'),
                    ], className='col-md-12'),

                    html.Div([
                        # classes parameter
                        # html.Div([
                        #     html.P("Number of classes in data", className="hyperparameter-title"),
                        #     dcc.Slider(
                        #         id='classes_picker',
                        #         min=2,
                        #         max=4,
                        #         step=1,
                        #         value=2,
                        #     ),
                        # ], className='slider'
                        # ),

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
                            html.H6([
                                html.Span("Number of samples in data:",
                                          title="Used to generate dataset of given number of rows")
                            ], className="hyperparameter-title"),
                            dcc.Slider(
                                id='samples_picker',
                                min=10,
                                max=1000,
                                step=5,
                                value=300,
                                tooltip={'always_visible': False, 'placement': 'bottomLeft'},
                                updatemode='drag',
                                marks={
                                    10: {'label': '10'},
                                    100: {'label': '100'},
                                    200: {'label': '200'},
                                    300: {'label': '300'},
                                    400: {'label': '400'},
                                    500: {'label': '500'},
                                    600: {'label': '600'},
                                    700: {'label': '700'},
                                    800: {'label': '800'},
                                    900: {'label': '900'},
                                    1000: {'label': '1000'},
                                }
                            ),
                        ], className='border slider bg-white p-2'),

                        # data generation button
                        html.Button(
                            id='data-generator',
                            children=['Generate'],
                            className="btn btn-info m-3"
                        ),

                        html.P(
                            "The data generated on the click of this button is random, yet follows a pattern to allow for proper classification by forming two major clusters (the two classes of the data). Since logistic regression classier works best of two-class classifications, we shall work with two classes. The graph offers a visualisation of the data, how it is clustered and its spread on the 2D plain. Next, we shall see how tweaking the various hyperparameters, as offered by the sklearn.linear_model.LogisticRegression library of the data, allows us to fine tune our classifier model and how that affects the resulting accuracy.",
                            className="m-2"),
                    ], className='col-md-6'),
                    # data graph
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='dataset_graph',
                                # figure='fig'
                            ),
                        ], className="border bg-white"),

                    ], className='col-md-6'),

                ], className='row'),

                html.Hr(className="m-4"),

                # upper half
                html.Div([
                    # tuners
                    html.Div([
                        html.H4('Tuners here'),

                        # training size

                        html.Div([

                            html.H6([html.Span("Training Size:", title="Specify Training Size Of Model")]),

                            dcc.Slider(
                                id='training_size_picker',
                                min=5,
                                max=95,
                                step=1,
                                value=training_size,
                                tooltip={'always_visible': False, 'placement': 'bottomLeft'},
                                updatemode='drag',
                                marks={
                                    5: {'label': '5'},
                                    10: {'label': '10'},
                                    15: {'label': '15'},
                                    20: {'label': '20'},
                                    25: {'label': '25'},
                                    30: {'label': '30'},
                                    35: {'label': '35'},
                                    40: {'label': '40'},
                                    45: {'label': '45'},
                                    50: {'label': '50'},
                                    55: {'label': '55'},
                                    60: {'label': '60'},
                                    65: {'label': '65'},
                                    70: {'label': '70'},
                                    75: {'label': '75'},
                                    80: {'label': '80'},
                                    85: {'label': '85'},
                                    90: {'label': '90'},
                                    95: {'label': '95'},
                                }
                            ),
                        ], className='border bg-white p-2'),

                        # solver
                        html.Div([
                            html.H6(
                                [html.Span("Solver parameter:", title="Algorithm to use in the optimization problem.")],
                                className="hyperparameter-title"),
                            dcc.Dropdown(
                                id='solver_picker',
                                options=solver_options,
                                value=solver_value
                            ),
                        ], className='border bg-white p-2 mt-2'
                        ),

                        # dual
                        html.Div([
                            html.H6([html.Span("Dual parameter:",
                                               title="Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.")],
                                    className="hyperparameter-title"),

                            html.Div([], id='dual_picker_div')
                            # dcc.RadioItems(
                            #     id='dual_picker',
                            #     options=[
                            #         {'label': 'True', 'value': 'True'},
                            #         {'label': 'False', 'value': 'False'},
                            #     ],
                            #     value='False'
                            # ),
                        ], className="border bg-white p-2 mt-2", id='dual-div'),
                        # if dual_show else html.H1('Dual here', id='dual_picker', ),

                        # C parameter
                        html.Div([
                            html.H6([html.Span("C parameter:",
                                               title="Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.")],
                                    className="hyperparameter-title"),
                            dcc.Slider(
                                id='c_picker',
                                min=0,
                                max=5,
                                step=0.1,
                                value=1.0,
                                tooltip={'always_visible': False, 'placement': 'bottomLeft'},
                                updatemode='drag',
                                marks={
                                    0: {'label': '0'},
                                    1: {'label': '1'},
                                    2: {'label': '2'},
                                    3: {'label': '3'},
                                    4: {'label': '4'},
                                    5: {'label': '5'},
                                }
                            ),
                        ], className='border bg-white p-2 mt-2'
                        ),

                        # fit_intercept
                        html.Div([
                            html.H6([html.Span("Fit intercept:",
                                               title="Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.")],
                                    className="hyperparameter-title"),
                            dcc.RadioItems(
                                id='fit_intercept_picker',
                                options=[
                                    {'label': 'True', 'value': 'True'},
                                    {'label': 'False', 'value': 'False'},
                                ],
                                value='True'
                            ),
                        ], className="border bg-white p-2 mt-2"),

                        # penalty
                        html.Div([
                            html.H6([
                                html.Span("Penalty parameter:",
                                          title="Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.")],
                                className="hyperparameter-title"),

                            html.Div([
                            ],
                                id='penalty_picker_div', ),
                            # dcc.Dropdown(
                            #     id='penalty_picker',
                            #     options=penalty_options,
                            #     value=penalty_value
                            # ),
                        ], className='border bg-white p-2 mt-2'
                        ),

                        # random_state parameter
                        html.Div([
                            html.H6([html.Span("Random State",
                                               title="Whenever randomization is part of a Scikit-learn algorithm, a random_state parameter may be provided to control the random number generator used. Note that the mere presence of random_state doesn’t mean that randomization is always used, as it may be dependent on another parameter, e.g. shuffle, being set.")],
                                    className="hyperparameter-title"),
                            html.Div([
                                # retrun updated random state picker here
                            ], id='random_state_div',
                            ),
                        ], className='border bg-white p-2 mt-2'
                        ),

                        # max_iter
                        html.Div([
                            html.H6([html.Span("Max Iter:",
                                               title="Maximum number of iterations taken for the solvers to converge.")],
                                    className="hyperparameter-title"),
                            dcc.Slider(
                                id='max_iter_picker',
                                min=1,
                                max=300,
                                step=1,
                                value=100,
                                marks={
                                    1: {'label': '1'},
                                    50: {'label': '50'},
                                    100: {'label': '100'},
                                    150: {'label': '150'},
                                    200: {'label': '200'},
                                    250: {'label': '250'},
                                    300: {'label': '300'}
                                }
                            ),
                        ], className='border bg-white p-2 mt-2'
                        ),

                        # multi_class
                        # html.Div([
                        #     html.P("multi class parameter", className="hyperparameter-title"),
                        #     dcc.Dropdown(
                        #         id='multi_class_picker',
                        #         options=multi_class_options,
                        #         value=multi_class_value
                        #     ),
                        # ], className='dropdown'
                        # ),

                        html.Button(
                            id='model-trainer',
                            children=['Train'],
                            className="btn btn-info mt-2"
                        ),

                    ],
                        className='col-md-6'),
                    # graph
                    html.Div([
                        html.Div([
                            dcc.Graph(
                                id='trained_model_graph',
                            ),
                        ], className="border"),
                    ],
                        className='col-md-6'),
                ], className='row'),

                html.Hr(className="m-4"),

                # lower half
                html.Div([
                    html.Div(
                        [
                            html.H4('Model Analysis'),

                            html.Div([], id='classification_score'),

                        ], className='bg-white col-md-6'
                    ),
                    html.Div(
                        [
                            # html.H4('Confusion Matrix'),
                            html.Div([
                                dcc.Graph(
                                    id='confusion_matrix',
                                )
                            ], className='heatmap')
                        ], className='border col-md-6'
                    ),
                ], className='row'),
            ], id="playground"),

        # Blank Dummy Elements
        html.Div([
            dcc.RadioItems(
                id=dynamic_id_dual,
                options=[
                    {'label': 'True', 'value': 'True', 'disabled': True},
                    {'label': 'False', 'value': 'False', 'disabled': True},
                ],
                value='False'
            ),
            dcc.Dropdown(
                id=dynamic_id_penalty_picker,
                options=penalty_options,
                value=penalty_value
            ),
            dcc.Slider(
                id=dynamic_id_random_state_picker,
                min=0,
                max=10,
                step=1,
                value=0,
                disabled=True,
            ),
        ], id='asd'),

    ], className='container'),

    html.Hr(),

    html.Footer([
        html.H6(
            [html.A("Github Repository", href="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/")]),
        html.H6("Contributors:"),
        html.Div([
            html.Div([
                html.Img(
                    src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/profile_photos/ritik.png?raw=true",
                    className="rounded-circle", style={"width": "40px", "height": "40px"}),
                html.A("Ritik Verma", href="https://www.linkedin.com/in/ritik-v-100516a5/", className="ml-2"),
            ], className="col-md-6 mt-2"),

            html.Div([
                html.Img(
                    src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/profile_photos/ritwik.jpg?raw=true",
                    className="rounded-circle", style={"width": "40px", "height": "40px"}),
                html.A("Ritwik Ghosh", href="https://www.linkedin.com/in/ritwik-ghosh-01ba01152/", className="ml-2"),
            ], className="col-md-6 mt-2"),

            html.Div([
                html.Img(
                    src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/profile_photos/priscila.jpeg?raw=true",
                    className="rounded-circle", style={"width": "40px", "height": "40px"}),
                html.A("Priscila Tamang Ghising", href="https://www.linkedin.com/in/priscila-tamang-ghising-a703b1174/",
                       className="ml-2"),
            ], className="col-md-6 mt-2"),

            html.Div([
                html.Img(
                    src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/profile_photos/gazal.jpeg?raw=true",
                    className="rounded-circle", style={"width": "40px", "height": "40px"}),
                html.A("Gazal Garg", href="https://www.linkedin.com/in/gazal-garg-073a2719b/", className="ml-2"),
            ], className="col-md-6 mt-2"),

            html.Div([
                html.Img(
                    src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/profile_photos/mushreqa.jpeg?raw=true",
                    className="rounded-circle", style={"width": "40px", "height": "40px"}),
                html.A("Mushrequa Nawaz", href="https://www.linkedin.com/in/mushrequa-nawaz-a3b1881a0/",
                       className="ml-2"),
            ], className="col-md-6 mt-2"),
        ], className='row')
    ], className="container border bg-light m-5")

], )

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([layout_main
              ], id='page-content')
])


# functions


@app.callback(Output('random_state_div', 'children'),
              [Input('solver_picker', 'value')])
def update_random_state_picker(solver):
    global solver_options
    global solver_value
    global random_state_value
    global dynamic_id_random_state_picker

    dynamic_id_random_state_picker = 'dummy3'

    if solver == 'sag' or solver == 'saga' or solver == 'lbfgs':
        return dcc.Slider(
            id='random_state_picker',
            min=0,
            max=10,
            step=1,
            value=random_state_value,
            marks={
                0: {'label': '0'},
                1: {'label': '1'},
                2: {'label': '2'},
                3: {'label': '3'},
                4: {'label': '4'},
                5: {'label': '5'},
                6: {'label': '6'},
                7: {'label': '7'},
                8: {'label': '8'},
                9: {'label': '9'},
                10: {'label': '10'},
            }
        )
    else:
        return dcc.Slider(
            id='random_state_picker',
            min=0,
            max=10,
            step=1,
            value=0,
            disabled=True,
            marks={
                0: {'label': '0'},
                1: {'label': '1'},
                2: {'label': '2'},
                3: {'label': '3'},
                4: {'label': '4'},
                5: {'label': '5'},
                6: {'label': '6'},
                7: {'label': '7'},
                8: {'label': '8'},
                9: {'label': '9'},
                10: {'label': '10'},
            }
        )


@app.callback(Output('dual_picker_div', 'children'),
              [Input('solver_picker', 'value'), Input('penalty_picker', 'value')])
def update_dual_picker(solver, penalty):
    global solver_options
    global solver_value
    global multi_class_options
    global multi_class_value
    global penalty_options
    global penalty_value
    global dual_value
    global c_value
    global fit_intercept_value
    global random_state_value
    global max_iter_value
    global dual_show
    global random_state_show
    global dynamic_id_dual

    if solver == 'liblinear' and penalty == 'l2':
        dynamic_id_dual = 'dummy1'
        return dcc.RadioItems(
            id='dual_picker',
            options=[
                {'label': 'True', 'value': 'True'},
                {'label': 'False', 'value': 'False'},
            ],
            value='False'
        ),
    else:
        dual_value = False
        dynamic_id_dual = 'dummy1'
        return dcc.RadioItems(
            id='dual_picker',
            options=[
                {'label': 'True', 'value': 'True', 'disabled': True},
                {'label': 'False', 'value': 'False', 'disabled': True},
            ],
            value='False'
        ),


@app.callback(Output('penalty_picker_div', 'children'),
              [Input('solver_picker', 'value')])
def update_penalty_picker(solver):
    global solver_options
    global solver_value
    global multi_class_options
    global multi_class_value
    global penalty_options
    global penalty_value
    global dual_value
    global c_value
    global fit_intercept_value
    global random_state_value
    global max_iter_value
    global dual_show
    global random_state_show
    global dynamic_id_penalty_picker

    dynamic_id_penalty_picker = 'dummy2'

    # penalty
    if solver == 'newton-cg' or solver == 'sag' or solver == 'lbfgs':
        penalty_options = [
            {
                'label': 'l2', 'value': 'l2',
            },
        ]
        if penalty_value != 'l2':
            penalty_value = 'l2'

        return dcc.Dropdown(
            id='penalty_picker',
            options=penalty_options,
            value=penalty_value
        )

    elif solver != 'saga':
        penalty_options = [
            {
                'label': 'l1', 'value': 'l1',
            },
            {
                'label': 'l2', 'value': 'l2',
            },
            {
                'label': 'none', 'value': 'none',
            },
        ]
        if penalty_value == 'elasticnet':
            penalty_value = 'l2'

        return dcc.Dropdown(
            id='penalty_picker',
            options=penalty_options,
            value=penalty_value
        )

    elif solver != 'liblinear':
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
        ]
        if penalty_value == 'none':
            penalty_value = 'l2'

        return dcc.Dropdown(
            id='penalty_picker',
            options=penalty_options,
            value=penalty_value
        )

    else:
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
                'label': 'none', 'value': 'none',
            },
        ]

        return dcc.Dropdown(
            id='penalty_picker',
            options=penalty_options,
            value=penalty_value
        )


@app.callback(Output('asd', 'style'),
              [Input('samples_picker', 'value')])
def set_data(samples):
    # print('set data')
    # global n_clusters
    global n_classes
    global n_samples
    global isDataSet
    n_classes = 2
    # n_clusters = clusters
    n_samples = samples

    # isDataSet = True
    # print('set data 2')
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
    global firstGeneration

    if not firstGeneration:
        firstGeneration = True
        return go.Figure()

    # isDataGenerated = False
    # if not isDataSet:
    #     n_classes = 2
    #     n_samples = 300
    #     isDataSet = True

    isDataGenerated = False

    # print('generate data')
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
    # print('generate data 2')
    fig = go.Figure(data=traces)
    return fig.update_layout(
        title="Generated Data",
        yaxis=dict(
            range=(min(data.iloc[:, 1]) - 1, max(data.iloc[:, 1]) + 1),
            constrain='domain'
        )
    )


@app.callback([Output('trained_model_graph', 'figure'), Output('classification_score', 'children'),
               Output('confusion_matrix', 'figure'), ],
              [Input('model-trainer', 'n_clicks'), Input('penalty_picker', 'value'), Input('dual_picker', 'value'),
               Input('c_picker', 'value'), Input('fit_intercept_picker', 'value'),
               Input('random_state_picker', 'value'), Input('solver_picker', 'value'),
               Input('max_iter_picker', 'value'),
               Input('training_size_picker', 'value')])
def train_model(clicks, penalty, dual, c, fit_intercept, random_state, solver, max_iter, train_size):
    # if clicks == 0:
    #     return
    # global n_clusters
    global n_classes
    global n_samples
    global data
    global isDataGenerated
    global firstTraining
    global solver_value
    # global multi_class_value
    global penalty_value
    global dual_value
    global c_value
    global fit_intercept_value
    global random_state_value
    global max_iter_value
    global training_size

    # print("solver_value=", solver_value)
    # print("solver=", solver)
    solver_value = solver
    # print('value update')
    penalty_value = penalty
    dual_value = dual
    fit_intercept_value = fit_intercept
    random_state_value = random_state
    max_iter_value = max_iter
    training_size = train_size

    if not isDataGenerated:
        # firstTraining = True
        return go.Figure(), '', go.Figure()

    # if not isDataGenerated:
    #     return

    # print(random_state_show)
    # set_dependencies()
    # print(random_state_show)
    # print('training model')
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:2], data.iloc[:, 2],
                                                        test_size=(100 - training_size) / 100)

    model = LogisticRegression(penalty=penalty_value,
                               dual=True if dual_value == 'True' else False,
                               C=c_value,
                               fit_intercept=True if fit_intercept_value == 'True' else False,
                               random_state=random_state_value, solver=solver_value, max_iter=max_iter_value,
                               # multi_class='auto',
                               warm_start=False, n_jobs=-1 if solver_value != 'liblinear' else None, )
    trained_model = model.fit(x_train, y_train)
    # prediction_train = trained_model.predict(x_train)
    prediction_test = trained_model.predict(x_test)
    # prediction_total = trained_model.predict(data.iloc[:, 0:2])
    coef = trained_model.coef_
    # print(trained_model.intercept_)
    # y = coef[0][0] * data[0] + coef[0][1]
    y = -(data[0] * coef[0][0] + trained_model.intercept_[0]) / coef[0][1]

    traces = []
    for i in range(0, n_classes):
        bool_cond = y_train == i
        bool_cond_2 = prediction_test == i
        # print(x_train[bool_cond])
        color = "red"
        if (i == 1):
            color = "purple"

        traces.append(go.Scatter(x=x_train[bool_cond][0], y=x_train[bool_cond][1], mode='markers',
                                 name='class ' + str(i) + ' training data', marker_symbol='triangle-up',
                                 marker_color=color))
        traces.append(go.Scatter(x=x_test[bool_cond_2][0], y=x_test[bool_cond_2][1], mode='markers',
                                 name='class ' + str(i) + ' test data', marker_color=color))

    traces.append(go.Scatter(x=data[0], y=y, mode='lines', name="Regression Line"))

    # confusion martix generation

    matrix = confusion_matrix(y_test, prediction_test)

    # classification report generation

    classification = classification_report(y_test, prediction_test)

    result = []
    classification = classification.split("\n")

    for i in [classification[i] for i in [2, 3, 5, 6, 7]]:
        result.append(i.split())

    # print(classification)
    # print(trained_model.coef_, trained_model.intercept_)
    fig = go.Figure(data=traces)
    return fig.update_layout(title="Trained Model",
                             yaxis=dict(
                                 range=(min(data.iloc[:, 1]) - 1, max(data.iloc[:, 1]) + 1),
                                 constrain='domain'
                             )), html.Table([
        html.Tr([
            html.Th(),
            html.Th("Precision"),
            html.Th("Recall"),
            html.Th("F1-Score"),
            html.Th("Support"),
        ]),
        html.Tr([
            html.Th("0"),
            html.Td(result[0][1]),
            html.Td(result[0][2]),
            html.Td(result[0][3]),
            html.Td(result[0][4]),
        ]),
        html.Tr([
            html.Th("1"),
            html.Td(result[1][1]),
            html.Td(result[1][2]),
            html.Td(result[1][3]),
            html.Td(result[1][4]),
        ]),
        html.Tr([
            html.Th("Accuracy"),
            html.Td(),
            html.Td(),
            html.Td(result[2][1]),
            html.Td(result[2][2]),
        ]),
        html.Tr([
            html.Th("Macro Avg"),
            html.Td(result[3][2]),
            html.Td(result[3][3]),
            html.Td(result[3][4]),
            html.Td(result[3][5]),
        ]),
        html.Tr([
            html.Th("Weighted Avg"),
            html.Td(result[4][2]),
            html.Td(result[4][3]),
            html.Td(result[4][4]),
            html.Td(result[4][5]),
        ]),
    ], className='table table-hover table-sm'), go.Figure(data=[{
        "type": "heatmap",
        "x": ["Predicted 0's", "Predicted 1's"],
        "y": ["Actual 0's", "Actual 1's"],
        "z": matrix
    }], layout={"title": "Confusion Matrix"})


def set_dependencies():
    global solver_options
    global solver_value
    global multi_class_options
    global multi_class_value
    global penalty_options
    global penalty_value
    global dual_value
    global c_value
    global fit_intercept_value
    global random_state_value
    global max_iter_value
    global dual_show
    global random_state_show

    # penalty
    if solver_value == 'newton-cg' or solver_value == 'sag' or solver_value == 'lbfgs':
        penalty_options = [
            {
                'label': 'l2', 'value': 'l2',
            },
        ]
        if penalty_value != 'l2':
            penalty_value = 'l2'

    elif solver_value != 'saga':
        penalty_options = [
            {
                'label': 'l1', 'value': 'l1',
            },
            {
                'label': 'l2', 'value': 'l2',
            },
            {
                'label': 'none', 'value': 'none',
            },
        ]
        if penalty_value == 'elasticnet':
            penalty_value = 'l2'

    elif solver_value != 'liblinear':
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
        ]
        if penalty_value == 'none':
            penalty_value = 'l2'
    else:
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
                'label': 'none', 'value': 'none',
            },
        ]

    # dual
    if solver_value == 'liblinear' and penalty_value == 'l2':
        dual_show = True
    else:
        dual_show = False
        dual_value = False

    # random state
    if solver_value == 'sag' or solver_value == 'saga' or solver_value == 'lbfgs':
        random_state_show = True
        # print('changing status')
    else:
        random_state_show = False
        random_state_value = 0

    # multi class
    if solver_value == 'liblinear':
        multi_class_options = [
            {
                'label': 'auto', 'value': 'auto',
            },
            {
                'label': 'ovr', 'value': 'ovr',
            },
        ]
        if multi_class_value == 'multinomial':
            multi_class_value = 'auto'
    else:
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


# seq_count = 0
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    # print(pathname)
    if pathname == '/':
        return layout_main
    elif pathname == '/tour':
        # if(sqe_count == 0):
        #     tour.generate_sequence()
        #     seq_count = seq_count + 1
        return tour.layout_tour
    else:
        return '404'


if __name__ == "__main__":
    app.run_server(debug=False,
                   dev_tools_ui=False, dev_tools_props_check=False
                   )
