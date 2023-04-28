from flask import Flask, render_template, request
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
from bokeh.models import ColumnDataSource, Tabs, TabPanel, CategoricalColorMapper, CustomJS, Select
from bokeh.models.widgets import Toggle
from bokeh.embed import components
import itertools
from bokeh.palettes import Category20, Category10
from bokeh.models import LegendItem
from bokeh.models import LegendItem, Legend, GroupFilter, CDSView, Circle, CustomJSHover, HoverTool, Div
from bokeh.layouts import column, row, Spacer
from bokeh.events import MouseMove
import itertools
from flask import url_for
import os


def create_app():

    app = Flask(__name__)

    data = pd.read_csv("data/data.csv")

    @app.route('/')
    def home():
        img_folder = os.path.join('static', 'img', 'carousel')
        img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        return render_template('home.html', img_files=img_files)
    
    @app.route('/ourProject/')
    def ourProject():
        return render_template('ourProject.html')
    
    @app.route('/aboutUs/')
    def aboutUs():
        return render_template('aboutUs.html')
    
    @app.route('/contactUs/')
    def contactUs():
        return render_template('contactUs.html')

    DOSSIER_DATA = "data/csv"

    # Obtenez tous les fichiers CSV dans le dossier de données
    fichiers_data = [fichier for fichier in os.listdir(
        DOSSIER_DATA) if fichier.endswith('.csv')]

    # Obtenez tous les modèles, tâches et réductions de dimensions uniques dans les noms de fichiers
    DATASETS = sorted(list(set([fichier.split('_')[0] for fichier in fichiers_data])))
    print(DATASETS)
    TACHES = list(set([fichier.split('_')[1] for fichier in fichiers_data]))
    REDUCTIONS = list(set([fichier.split('_')[2].split('.')[0]
                           for fichier in fichiers_data]))

    @app.route('/yourExploration/', methods=['GET','POST'])  # /<str:reduc>/
    def yourExploration():
        if request.method == 'POST':

            dataset = request.form['dataset']
            tache = request.form['tache']
            reduction = request.form['reduction']

            chemin_fichier = os.path.join(
                DOSSIER_DATA, f"{dataset}_{tache}_{reduction}.csv")
            data = pd.read_csv(chemin_fichier)

            plot = create_plot(data,tache,reduction)
            script, div = components(plot)

            #tab4 = TabPanel(child=plot4, title="A BRAND NEW WORLD")
            #plot = Tabs(tabs=[tab1, tab2, tab3, tab4]) 

            return render_template('yourExploration.html', div=div, script=script, actual_dataset = dataset, actual_tache = tache, actual_reduction = reduction, DATASETS=DATASETS, TACHES=TACHES, REDUCTIONS=REDUCTIONS, fichiers_data=fichiers_data)
        else:
             return render_template('yourExploration.html', DATASETS=DATASETS, TACHES=TACHES, REDUCTIONS=REDUCTIONS,fichiers_data=fichiers_data)
   
    """ 
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')
    """
    
    def create_plot(data,tache,reduction):

        data = data.sample(frac=1)
        data.rename(columns={'feature_1': 'x', 'feature_2': 'y'}, inplace=True)

        classes = [colonne.split('_name')[0] for colonne in data.columns if '_name' in colonne]
        data.rename(columns=lambda col: f"{col.split('_name')[0]}" if col.split('_name')[0] in classes else col, inplace=True)

        print(data.columns)
        #data = data[data[classes].notna()]
        data.dropna(subset=classes, inplace=True)

        classes_sets = [data[col].unique() for col in classes]
        print(classes)
        print(classes_sets)
        
        """
        for task in classes:
            globals()[f"list_{task}"] = data[task].unique().tolist()
        """   

        color_pals = [list(itertools.islice(itertools.cycle(Category20[20]), len(classes))) for classes in classes_sets]

        color_mappings = []
        for i, col in enumerate(classes):
            # Créer le mapping de couleur pour chaque ensemble de classes
            color_mapping = CategoricalColorMapper(factors=classes_sets[i], palette=color_pals[i])
            color_mappings.append(color_mapping)
            print(color_mapping)
        
        max_size = 2000  # on peut prendre une portion du dataframe car on a shuffle
        data  = data[:max_size]

        plot_figure = figure(
            title= reduction.upper() +' projection of the dataset',
            width=900,
            height=600,
            tools='pan, wheel_zoom, reset, box_zoom, box_select, lasso_select, crosshair, tap, save'
        )

        zoom_figure = figure(
            width=300,
            height=300,
            x_axis_type=None,
            y_axis_type=None,
            tools='', #crosshair ou pas ?
            match_aspect=True,
            title='Zoomed view',
            visible=False, 
        )

        toggle_button = Toggle(
            label="Enable/Disable zoom",
            button_type="primary",
            active=True
        )

        toggle_callback = CustomJS(args=dict(zoom_figure=zoom_figure), code="""
            zoom_figure.visible = !zoom_figure.visible;
            zoom_figure.change.emit();
        """)

        toggle_button.js_on_click(toggle_callback)

        """
        <div>
            <div>
                <span style='font-size: 12px;'><b>class1:</b> @class1 </span>
            </div>
            <div>
                <img src='../static/images/wikiart/data/@name' style='float: left; aspect-ratio: auto;margin: 5px 5px 5px 5px;width:250px;'/>
            </div>
        </div>
        """

        hover_tool = HoverTool(tooltips="""
            <div>
                <img src='../static/images/wikiart/data/@name' style='float: left; aspect-ratio: auto;margin: 5px 5px 5px 5px;width:250px;'/>
            </div>
        """)

        plot_figure.add_tools(hover_tool)


        legends = []
        for i, col in enumerate(classes):
            # Créer la légende pour chaque ensemble de classes
            legend = Legend(title=col, location="top_left",
                            title_text_font_style="bold italic", background_fill_alpha=0.6, visible=True)
            legend_items = []
            for clas, color in zip(classes_sets[i], color_pals[i]):
                source_subset = ColumnDataSource(data[data[col] == clas])
                glyph = plot_figure.circle("x", "y", source=source_subset,  fill_color={'field': col, 'transform': color_mappings[i]}, line_color='white',
                                        size=10, fill_alpha=0.8,visible = True) #color=color
                zoom_glyph = zoom_figure.circle("x", "y", source=source_subset, fill_color={'field': col, 'transform': color_mappings[i]}, line_color={'field': col, 'transform': color_mappings[i]},visible = True, line_width = 2, size=10, fill_alpha=0.7, line_alpha=0.8)
                if col != tache :
                    glyph.visible = False
                    zoom_glyph.visible = False
                legend_items.append(LegendItem(label=clas, renderers=[glyph]))
            legend.items = legend_items
            plot_figure.add_layout(legend)
            
            legends.append(legend)
        
        # Afficher uniquement la légende de la classe sélectionnée
        selected_class = tache 
        for legend in legends:
            if legend.title == selected_class:
                legend.visible = True
            else:
                legend.visible = False


        plot_figure.legend.click_policy="mute"
      

        """                         
        # Ajouter les légendes à la figure
        plot_figure.add_layout(legend1)
        plot_figure.add_layout(legend2)
        plot_figure.legend.click_policy = "mute"
        """
        Selecthandler = CustomJS(args=dict(plot_figure=plot_figure, legends=legends),
                         code="""
                            var value = cb_obj.value;
                            for (var j = 0; j < legends.length; j++) {
                                var legend_items = legends[j].items;
                                if (value == legends[j].title) {
                                    legends[j].visible = true;
                                    for (var i = 0; i < legend_items.length; i++) {
                                        legend_items[i].renderers[0].visible = true;
                                        legend_items[i].visible = true;
                                    }
                                } else {
                                    legends[j].visible = false;
                                    for (var i = 0; i < legend_items.length; i++) {
                                        legend_items[i].renderers[0].visible = false;
                                        legend_items[i].visible = false;
                                    }
                                }
                            }
                            plot_figure.change.emit();
                        """)



        

        update_zoom_callback = CustomJS(args=dict(plot_figure=plot_figure, zoom_figure=zoom_figure), code="""
        const x = cb_obj.x;
        const y = cb_obj.y;
        var zoom_range = 1;
        var value = cb_obj.value;
        var xs = plot_figure.x_range.start;
        var xe = plot_figure.x_range.end;
        var ys = plot_figure.y_range.start;
        var ye = plot_figure.y_range.end;
        var zoom_range_x = (xe - xs) /20;
        var zoom_range_y = (ye - ys) /20;

        zoom_figure.x_range.start = x - zoom_range_x;
        zoom_figure.x_range.end = x  + zoom_range_x;
        zoom_figure.y_range.start = y - zoom_range_y;
        zoom_figure.y_range.end = y + zoom_range_y;


        //zoom_figure.x_range.start = x - zoom_range;
        //zoom_figure.x_range.end = x  + zoom_range;
        //zoom_figure.y_range.start = y - zoom_range;
        //zoom_figure.y_range.end = y + zoom_range;
        """)

        plot_figure.js_on_event(MouseMove, update_zoom_callback)

        """
        select = Select(title="Option:", value="classes1",
                        options=["classes1", "classes2"])
        select.js_on_change("value", Selecthandler)
        """
        select = Select(title="Option:", value=tache,
                options=classes)
        
        select.js_on_change("value", Selecthandler)

        select.js_on_change("value", update_zoom_callback)

        layout = column(
            select,
            row(
                plot_figure,
                Spacer(width=30),
                column(
                    Spacer(height=30),
                    toggle_button,
                    zoom_figure,
                ),
            ),
        )

        return layout

    def create_plot1(array):

        max_size = 2000
        colors = itertools.cycle(Category20[20])
        pal = [color for m, color in zip(range(len(classes)), colors)]
        data_df = pd.DataFrame(array[:max_size], columns=('x', 'y'))
        data_df['class'] = [x for x in labels][:max_size]
        datasource = ColumnDataSource(data_df)
        color_mapping = CategoricalColorMapper(factors=classes, palette=pal)
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

    """   
    umap_2d = umap.umap_.UMAP(n_components=2)
    umap_2d.fit(features)
    projections_umap = umap_2d.transform(features) 
    """

    return app