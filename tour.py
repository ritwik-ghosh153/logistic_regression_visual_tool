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
import os
import pickle


def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.   
    """
    pos = y == 1
    neg = y == 0

    traces = []

    traces.append(go.Scatter(x=X[pos, 0], y=X[pos, 1], mode='markers', name='Marks 1'))
    traces.append(go.Scatter(x=X[neg, 0], y=X[neg, 1], mode='markers', name='Marks 2'))

    return traces


data = pd.read_csv("Dataset/marks.txt")
X = data.iloc[:, 0:-1].values
y = data.iloc[:, -1].values

plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

pos = y == 1
neg = y == 0

class_1_data = go.Scatter(x=X[pos, 0], y=X[pos, 1], mode='markers', name='Marks 1')
class_2_data = go.Scatter(x=X[neg, 0], y=X[neg, 1], mode='markers', name='Marks 2')

# lines = [[-0.06939174423370675, 0.010906847944679096, 0.0009908164877343582], [-0.13832469007556072, 0.011385771514523729, 0.0015268789001446138], [-0.2066655098772045, 0.011861130890198199, 0.0020586162220766566], [-0.2744192178469147, 0.012332951548887397, 0.0025860531092747998], [-0.34159090667045294, 0.01280125948838644, 0.0031092152496578745], [-0.4081857407809602, 0.013266081186049881, 0.0036281293166576392], [-0.47420894977052397, 0.013727443558927025, 0.004142822921980422], [-0.5396658219548695, 0.014185373925123796, 0.004653324567973912], [-0.6045616981013372, 0.014639899966423139, 0.00515966359977417], [-0.6689019653290108, 0.015091049692187794, 0.005661870157399383], [-0.7326920511886524, 0.015538851404562312, 0.0061599751279487565], [-0.7959374179288676, 0.01598333366498381, 0.006654010098055069], [-0.8586435569538451, 0.016424525262005524, 0.007144007306730524], [-0.9208159834768388, 0.01686245518043076, 0.0076299995987348075], [-0.9824602313726387, 0.017297152571750452, 0.008112020378585147], [-1.0435818482312043, 0.01772864672587269, 0.00859010356531712], [-1.1041863906137748, 0.01815696704412816, 0.009064283548095402], [-1.1642794195119779, 0.01858214301353296, 0.009534595142763929], [-1.2238664960095442, 0.019004204182285687, 0.01000107354941422], [-1.282953177145631, 0.019423180136474087, 0.010463754311041873], [-1.3415450119780807, 0.019839100477964004, 0.010922673273352038], [-1.399647537844302, 0.02025199480344114, 0.011377866545765114], [-1.4572662768169355, 0.020661892684574662, 0.011829370463665934], [-1.514406732351065, 0.02106882364927095, 0.01227722155193192], [-1.5710743861191563, 0.021472817163983752, 0.012721456489766969], [-1.6272746950297134, 0.021873902617047306, 0.013162112076862173], [-1.6830130884251644, 0.022272109302997754, 0.013599225200896334], [-1.7382949654543238, 0.0226674664078484, 0.014032832806384121], [-1.7931256926144783, 0.023060002995283775, 0.014462971864873218], [-1.8475106014580078, 0.023449747993737943, 0.014889679346487153], [-1.9014549864582748, 0.02383673018432266, 0.015312992192805214], [-1.9549641030293716, 0.024220978189570707, 0.015732947291066543], [-2.0080431656943323, 0.024602520462961346, 0.016149581449682145], [-2.060697346396233, 0.024981385279194045, 0.01656293137503408], [-2.112931772946695, 0.02535760072517833, 0.016973033649538852], [-2.1647515276062705, 0.02573119469170795, 0.01737992471094911], [-2.216161645791165, 0.026102194865787794, 0.01778364083286491], [-2.26716711490085, 0.026470628723583516, 0.01818421810642414], [-2.317772873261233, 0.026836523523964737, 0.0185816924231407], [-2.3679838091779652, 0.027199906302612376, 0.018976099458855963], [-2.4178047600947963, 0.02756080386666357, 0.019367474658769947], [-2.467240511851654, 0.0279192427898656, 0.019755853223515283], [-2.5162957980377256, 0.02827524940821522, 0.020141270096239797], [-2.564975299434395, 0.02862884981605616, 0.020523759950659297], [-2.613283643543471, 0.028980069862612295, 0.020903357180045122], [-2.6612254041959797, 0.02932893514893228, 0.021280095887108788], [-2.7088051012371626, 0.029675471025224248, 0.0216540098747478], [-2.7560272002832322, 0.03001970258855824, 0.02202513263761538], [-2.8028961125458194, 0.030361654680916568, 0.02239349735447872], [-2.849416194720019, 0.030701351887572087, 0.02275913688132983], [-2.895591748932172, 0.031038818535775482, 0.023122083745213986], [-2.941427022743712, 0.03137407869373406, 0.023482370138741958], [-2.9869262092073416, 0.031707156169863526, 0.023840027915251225], [-3.032093446972314, 0.03203807451229791, 0.02419508858458499], [-3.0769328204354114, 0.03236685700864096, 0.024547583309456174], [-3.1214483599344733, 0.032693526685943886, 0.024897542902365205], [-3.1656440419815155, 0.033018106310895474, 0.025244997823042053], [-3.2095237895325903, 0.033340618390211206, 0.02558997817638321], [-3.2530914722916093, 0.03366108517120767, 0.025932513710855292], [-3.296350907045574, 0.03397952864255063, 0.026272633817338308], [-3.339305858028727, 0.034295970535164436, 0.02661036752838227], [-3.381960037313297, 0.03461043232329214, 0.026945743517852082], [-3.4243171052245787, 0.034922935225695194, 0.027278790100936322], [-3.4663806707783125, 0.035233500206983306, 0.027609535234497194], [-3.5081542921382614, 0.03554214797906407, 0.02793800651773851], [-3.5496414770921842, 0.03584889900270399, 0.028264231193171337], [-3.590845683544395, 0.0361537734891921, 0.028588236147856456], [-3.6317703200232456, 0.03645679140209819, 0.028910047914904542], [-3.67241874620185, 0.036757972459117334, 0.02922969267521483], [-3.7127942734306854, 0.03705733613399424, 0.029547196259435464], [-3.7529001652806175, 0.03735490165852036, 0.029862584150128643], [-3.792739638094902, 0.037650688024596285, 0.030175881484123333], [-3.83231586154918, 0.03794471398635514, 0.030487113055042685], [-3.871631959218042, 0.03823699806233937, 0.03079630331598961], [-3.9106910091471816, 0.038527558537726145, 0.03110347638237771], [-3.9494960444301364, 0.03881641346659658, 0.031408656034895], [-3.9880500537886454, 0.039103580674243486, 0.03171186572258793], [-4.026355982155631, 0.03938907775951275, 0.03201312856605348], [-4.064416731260161, 0.039672922097175004, 0.03231246736072992], [-4.10223516021341, 0.03995513084032237, 0.03260990458027423], [-4.139814086095044, 0.04023572092278728, 0.032905462380017896], [-4.177156284539303, 0.04051470906157962, 0.033199162600491265], [-4.21426449032016, 0.040792111759338646, 0.03349102677100798], [-4.251141397934941, 0.04106794530679642, 0.033781076113301334], [-4.287789662185939, 0.04134222578525007, 0.03406933154520517], [-4.3242118987595175, 0.04161496906904028, 0.034355813684372356], [-4.360410684802215, 0.04188619082803307, 0.0346405428520239], [-4.3963885594934125, 0.04215590653010206, 0.03492353907672224], [-4.432148024614276, 0.04242413144361016, 0.03520482209816375], [-4.467691545112544, 0.042690880639887484, 0.03548441137098431], [-4.503021549662807, 0.04295616899570357, 0.03576232606857269], [-4.538140431222181, 0.04322001119573343, 0.03603858508688855], [-4.573050547580753, 0.04348242173501318, 0.036313207048277854], [-4.607754221907008, 0.04374341492138671, 0.03658621030528512], [-4.64225374328766, 0.04400300487793966, 0.03685761294445595], [-4.676551367261769, 0.044261205545419485, 0.03712743279012693], [-4.710649316349198, 0.04451803068464241, 0.037395687408201324], [-4.744549780573004, 0.04477349387888403, 0.03766239410990572], [-4.778254917975628, 0.045027608536252386, 0.03792756995552495]]

lines = [[8.750000000000001e-05, 0.010914227450795143, 0.01042672829727844],
         [-1.2801450454173464, 0.01744523230895539, 0.011651651607577458],
         [-2.3620954755099226, 0.025646142015803217, 0.01978977936584616],
         [-3.287798066038562, 0.03276687291223758, 0.02678186559361695],
         [-4.091632010265211, 0.03902616226774608, 0.032867931727349274]]

report = [[['0', '0.00', '0.00', '0.00', '0'], ['1', '1.00', '0.65', '0.79', '20'], ['accuracy', '0.65', '20'],
           ['macro', 'avg', '0.50', '0.33', '0.39', '20'], ['weighted', 'avg', '1.00', '0.65', '0.79', '20']],
          [['False', '0.14', '1.00', '0.25', '1'], ['True', '1.00', '0.68', '0.81', '19'], ['accuracy', '0.70', '20'],
           ['macro', 'avg', '0.57', '0.84', '0.53', '20'], ['weighted', 'avg', '0.96', '0.70', '0.78', '20']],
          [['False', '0.57', '1.00', '0.73', '4'], ['True', '1.00', '0.81', '0.90', '16'], ['accuracy', '0.85', '20'],
           ['macro', 'avg', '0.79', '0.91', '0.81', '20'], ['weighted', 'avg', '0.91', '0.85', '0.86', '20']],
          [['False', '0.86', '1.00', '0.92', '6'], ['True', '1.00', '0.93', '0.96', '14'], ['accuracy', '0.95', '20'],
           ['macro', 'avg', '0.93', '0.96', '0.94', '20'], ['weighted', 'avg', '0.96', '0.95', '0.95', '20']],
          [['False', '0.86', '1.00', '0.92', '6'], ['True', '1.00', '0.93', '0.96', '14'], ['accuracy', '0.95', '20'],
           ['macro', 'avg', '0.93', '0.96', '0.94', '20'], ['weighted', 'avg', '0.96', '0.95', '0.95', '20']]]


def make_line(theta):
    global class_1_data
    global class_2_data
    global plot_x

    plot_y = ((-1) / theta[2]) * (theta[1] * plot_x + theta[0])

    plot = go.Scatter(x=plot_x, y=plot_y, mode='lines', name="Regression Line")
    return [plot, class_1_data, class_2_data]


sample_lines = []
count = 0
count_2 = 0
for i, result in zip(lines, report):
    # print(count)
    count = count + 1
    if count == 1:
        layout = go.Layout(
            autosize=False,
            title="Initial Regression Line",
            yaxis=dict(
                range=(0, 120),
                constrain='domain'
            )
        )
        count_2 = count_2 + 20000
        sample_lines.append(
            html.Div([
                html.Div([
                    html.Div([
                        dcc.Graph(
                            figure=go.Figure(data=make_line(i), layout=layout
                                             )
                        )], className="col-md-12"),

                    html.Div([
                        html.Table([
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
                        ], className='table table-hover table-sm')
                    ], className="col-md-12 table-responsive bg-white mt-3"),
                ], className="row")
            ], className="carousel-item container active"))
    else:

        layout = go.Layout(
            autosize=False,
            title="After " + str(count_2) + " iterations",
            yaxis=dict(
                range=(0, 120),
                constrain='domain'
            )
        )
        count_2 = count_2 + 20000
        sample_lines.append(
            html.Div([
                html.Div([
                    html.Div(
                        [
                            dcc.Graph(
                                figure=go.Figure(data=make_line(i), layout=layout
                                                 )
                            )], className="col-md-12"),
                    html.Div([
                        html.Table([
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
                        ], className='table table-hover table-sm')
                    ], className="col-md-12 table-responsive bg-white mt-3"),
                ], className="row")

            ], className="carousel-item container"))

layout_tour = html.Div([

    # NAV BAR
    html.Nav([
        html.A("Playground", className="flex-sm-fill text-sm-center nav-link text-white", href="/"),
        html.A("Tour", className="flex-sm-fill text-sm-center nav-link bg-info text-white", href="/tour")
    ], className="nav nav-pills flex-column flex-sm-row bg-dark"),

    html.Div(children=
    [
        html.H1('Logistic Regression Tour', className="text-center"),

        html.Hr(),

        html.P(
            "Before jumping into what LOGISTIC REGRESSION is let’s have a basic understanding of what exactly a “Linear Regression “ is. So, basically Linear Regression is a statistical method which is used to find an equation that can predict an outcome for a binary variable i.e Y, based on one or more response variables i.e X. Also, the response variable in linear regression strictly requires continuous data."),
        html.P(
            "Similarly, Logisitc Regression is also same as Linear regression which is used to find an equation to predict an outcome based on response variables but the only difference is that the model here does not strictly require continuous data, it can be both categorical or continuous. Logistic Regression can be used for various classification problems such as spam detection, Diabetes prediction,  whether the user will click on a given advertisement link or not and many more. To fit the final model, Logistic Regression uses an “iterative maximum likelihood” method rather than a least squares and “log odds ratio” rather than probabilities. Therefore, these methods are more appropriate for non normally distributed data which gives more freedom to researcher."),
        html.P(
            " Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables."),

        html.Hr(),

        html.H4("The general Linear Regression Equation is:"),

        html.Img(
            src="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/blob/develop/images/equation.png?raw=true",
            className="rounded mx-auto d-block"),
        html.P("Where, y is dependent variable and x1, x2 ... and Xn are explanatory variables."),

        html.Hr(),

        html.H4("Lets understand how it works!Lets understand how it works!"),
        html.P(
            "Basically Logistic Regression uses logistic function which is a “Sigmoid Function” that gives an ‘S’ shaped curve by taking any real input value and giving an output of a value between zero and one. "),
        html.P("Some of the conditions are:"),

        html.Ul([
            html.Li(
                "If the curve goes to positive infinity, Y predicted will become 1 and if the curve goes to negative infinity, Y predicted will become 0."),
            html.Li(
                "Similarly, if the output is more than 0.5 then we can classify the outcome as 1 or YES and if the output is less than 0.5 then it is classified as 0 or NO. For example if the output is 0.75 then we can say that there is 75 percent chance that a student will pass the exam. ")
        ]),

        html.Hr(),

        html.H4("Basic diagram for logistic function:"),

        html.Img(src="https://miro.medium.com/max/640/1*OUOB_YF41M-O4GgZH_F2rw.png",
                 className="rounded mx-auto d-block"),

        html.Hr(),

        html.H2("Showing logistic regression using gradient descent approach:"),

        html.Div([

            html.Div(sample_lines, className="carousel-inner"),

            html.A([
                html.Span(className="carousel-control-prev-icon bg-dark p-3", **{"aria-hidden": "true"})
            ], className="carousel-control-prev", href="#graphs_showing_regression", role="button",
                **{"data-slide": "prev"}),

            html.A([
                html.Span(className="carousel-control-next-icon bg-dark p-3", **{"aria-hidden": "true"})
            ], className="carousel-control-next", href="#graphs_showing_regression", role="button",
                **{"data-slide": "next"})

        ], className="carousel slide mb-3", id="graphs_showing_regression", **{"data-ride": "carousel"})

    ], className='container mt-2'),

    html.Footer([
        
        html.Div([
            html.Div([html.H6([html.A("Github Repository", href="https://github.com/ritwik-ghosh153/logistic_regression_visual_tool/")]),], className="col-12"),
            html.Div([html.H6("Contributors:"),], className="col-12"),
        ], className="row"),
        
        


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
    ], className="container border bg-light mt-5")
])
