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
                        html.P("Number of samples in data", className="hyperparameter-title"),
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
                    html.H1('Tuners here'),

                    # training size

                    html.Div([
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
                    ], className='slider'),

                    # penalty
                    html.Div([
                        html.P("Penalty parameter", className="hyperparameter-title"),

                        html.Div([
                        ],
                            id='penalty_picker_div', ),
                        # dcc.Dropdown(
                        #     id='penalty_picker',
                        #     options=penalty_options,
                        #     value=penalty_value
                        # ),
                    ], className='dropdown'
                    ),

                    # dual
                    html.Div([
                        html.P("Dual parameter", className="hyperparameter-title"),

                        html.Div([], id='dual_picker_div')
                        # dcc.RadioItems(
                        #     id='dual_picker',
                        #     options=[
                        #         {'label': 'True', 'value': 'True'},
                        #         {'label': 'False', 'value': 'False'},
                        #     ],
                        #     value='False'
                        # ),
                    ], className="radio", id='dual-div'),
                    # if dual_show else html.H1('Dual here', id='dual_picker', ),

                    # C parameter
                    html.Div([
                        html.P("C parameter", className="hyperparameter-title"),
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
                        html.Div([
                            # retrun updated random state picker here
                        ], id='random_state_div',
                        ),
                    ], className='slider'
                    ),

                    # solver
                    html.Div([
                        html.P("Solver parameter", className="hyperparameter-title"),
                        dcc.Dropdown(
                            id='solver_picker',
                            options=solver_options,
                            value=solver_value
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
                    ], className='slider'
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

                        html.Div([], id='classification_score'),

                    ], className='col-md-6'
                ),
                html.Div(
                    [
                        html.H1('Results here'),
                        html.Div([
                            dcc.Graph(
                                id='confusion_matrix',
                            )
                        ], className='heatmap')
                    ], className='col-md-6'
                ),
            ], className='row'),
        ], id="playground"),
    html.Div([], id='asd')
],
    className='container')


# functions


@app.callback(Output('random_state_div', 'children'),
              [Input('solver_picker', 'value')])
def update_random_state_picker(solver):
    global solver_options
    global solver_value
    global random_state_value

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

    if solver == 'liblinear' and penalty == 'l2':
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
                'label': 'none', 'value': 'none’',
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

    print("solver_value=", solver_value)
    print("solver=", solver)
    solver_value = solver
    print('value update')
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
                               # dual=eval(dual_value),
                               C=c_value,
                               # fit_intercept=fit_intercept_value,
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
        x = data[data['Labels'] == i]
        traces.append(go.Scatter(x=x[0], y=x[1], mode='markers', name='class ' + str(i)))

    traces.append(go.Scatter(x=data[0], y=y, mode='lines'))

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
    }])


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
                'label': 'none', 'value': 'none’',
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
                'label': 'none', 'value': 'none’',
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
        print('changing status')
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


if __name__ == "__main__":
    app.run_server(port=3000, debug=True, dev_tools_ui=False, dev_tools_props_check=False)
