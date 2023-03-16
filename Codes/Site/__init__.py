from flask import Flask, render_template
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
#!pip install umap-learn
#import umap
#import torch
#import plotly.express as px
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.embed import components

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def homepage():
        return render_template('homepage.html')

    # UMAP Réduction des features à 2 dimensions
    file = open("data/umap_array", "rb")
    #read the file to numpy array
    projections_umap = np.load(file, allow_pickle=True)
    #reduc=projections_umap

    @app.route('/Data/') #/<str:reduc>/
    def Data(reduc=projections_umap):

        plot=create_plot(reduc)
        script, div = components(plot)
        #show(plot_figure)

        return render_template('Data.html',plot=plot,div=div,script=script)
    
    @app.route('/Contact/')
    def Contact():
            return render_template('Contact.html')

    @app.route('/about/')
    def about():
        return render_template('about.html')

    @app.route('/hello/')
    @app.route('/hello/<name>')
    def hello(name='diallo'):
        return render_template('hello.html', name=name)

    @app.route('/plot.png')
    def plot_png():
        fig = create_figure()
        output = io.BytesIO()  
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')
    
    def create_figure():
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        xs = range(100)
        ys = [random.randint(1, 50) for x in xs]
        axis.plot(xs, ys)
        return fig
    
    def create_plot(array):
        max_size=2000
        data_df = pd.DataFrame(array[:max_size], columns=('x', 'y'))
        datasource = ColumnDataSource(data_df)
        plot_figure = figure(
            title='UMAP projection of the dataset',
            width=900,
            height=600,
            tools=('pan, wheel_zoom, reset'))
        plot_figure.circle(
            'x',
            'y',
            source=datasource)

        return plot_figure
    """   
    umap_2d = umap.umap_.UMAP(n_components=2)
    umap_2d.fit(features)
    projections_umap = umap_2d.transform(features) 
    """

    return app