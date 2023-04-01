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
from bokeh.models import ColumnDataSource, Tabs,TabPanel,CategoricalColorMapper, CustomJS, Select
from bokeh.embed import components
import itertools
from bokeh.palettes import Category20

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def homepage():
        return render_template('homepage.html')

    # UMAP Réduction des features à 2 dimensions
    UMAP = open("data/umap_array", "rb")
    PCA = open("data/pca_array_numpy.npy", "rb")
    TSNE = open("data/tsne_array_numpy.npy", "rb")
    cifar_labels=open("data/cifar_labels.npy", "rb")
    cifar_labels2=open("data/cifar_labels2.npy", "rb")

    #read the file to numpy array
    projections_UMAP = np.load(UMAP)
    #reduc=projections_umap
    projections_PCA=np.load(PCA)
    projections_TSNE=np.load(TSNE)
    labels=np.load(cifar_labels)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes2 = ('machina','animal')

    @app.route('/Data/') #/<str:reduc>/   
    def Data(reduc1=projections_PCA,reduc2=projections_TSNE,reduc3=projections_UMAP):

        plot1=create_plot1(reduc1)
        plot2=create_plot2(reduc2)
        plot3=create_plot3(reduc3)
        tab1 = TabPanel(child=plot1, title="PCA")
        tab2 = TabPanel(child=plot2, title="t-SNE")
        tab3 = TabPanel(child=plot3, title="UMAP")
        plot=Tabs(tabs=[tab1, tab2, tab3])
        script, div = components(plot)
        #show(plot_figure)

        return render_template('Data.html',div=div,script=script)
    
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
    
    def create_plot1(array):
        max_size=2000
        colors = itertools.cycle(Category20[20])    
        pal = [color for m, color in zip(range(len(classes)), colors)]
        data_df = pd.DataFrame(array[:max_size], columns=('x', 'y'))
        data_df['class'] = [x for x in labels][:max_size]
        datasource = ColumnDataSource(data_df)
        color_mapping = CategoricalColorMapper(factors=classes,palette=pal)
        plot_figure1 = figure(
            title='PCA projection of the dataset',
            width=900,
            height=600,
            tools=('pan, wheel_zoom, reset'))

        plot_figure1.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='class', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=10,
            legend_field="class"
        )
        plot_figure1.legend.location = "top_left"
        plot_figure1.legend.label_text_font_size = "20px"
        plot_figure1.title.text_font_size = '30pt'

        return plot_figure1

    def create_plot2(array):
        max_size=2000
        colors = itertools.cycle(Category20[20])    
        pal = [color for m, color in zip(range(len(classes)), colors)]
        data_df = pd.DataFrame(array[:max_size], columns=('x', 'y'))
        data_df['class'] = [x for x in labels][:max_size]
        datasource = ColumnDataSource(data_df)
        color_mapping = CategoricalColorMapper(factors=classes,palette=pal)
        plot_figure2 = figure(
            title='t-SNE projection of the dataset',
            width=900,
            height=600,
            tools=('pan, wheel_zoom, reset'))
        plot_figure2.circle(
            'x',
            'y',
            source=datasource,            
            color=dict(field='class', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=10,
            legend_field="class"
        )
        plot_figure2.legend.location = "top_left"
        plot_figure2.legend.label_text_font_size = "20px"
        plot_figure2.title.text_font_size = '30pt'


        return plot_figure2

    def create_plot3(array):
        max_size=2000
        colors = itertools.cycle(Category20[20])    
        pal = [color for m, color in zip(range(len(classes)), colors)]
        data_df = pd.DataFrame(array[:max_size], columns=('x', 'y'))
        data_df['class'] = [x for x in labels][:max_size]
        datasource = ColumnDataSource(data_df)
        color_mapping = CategoricalColorMapper(factors=classes,palette=pal)
        plot_figure3 = figure(
            title='UMAP projection of the dataset',
            width=900,
            height=600,
            tools=('pan, wheel_zoom, reset'))
        plot_figure3.circle(
            'x',
            'y',
            source=datasource,
            color=dict(field='class', transform=color_mapping),
            line_alpha=0.6,
            fill_alpha=0.6,
            size=10,
            legend_field="class"
        )
        plot_figure3.legend.location = "top_left"
        plot_figure3.legend.label_text_font_size = "20px"
        plot_figure3.title.text_font_size = '30pt'


        return plot_figure3

    """   
    umap_2d = umap.umap_.UMAP(n_components=2)
    umap_2d.fit(features)
    projections_umap = umap_2d.transform(features) 
    """

    return app