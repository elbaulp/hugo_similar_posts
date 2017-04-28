# coding: utf-8

# http://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
# http://brandonrose.org/clustering

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import mpld3
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity


def readPosts(path, english=False):
    """Read posts in path and return a pandas Data frame"""

    df = pd.DataFrame()
    titleRegEx = r'title ?[:=] ?"?([^"\n]*)'

    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as infile:
            txt = infile.read()
            title = re.search(titleRegEx, txt)
            data = [[os.path.basename(infile.name), txt, title.group(1)]]

            isEnglish = re.search('\.en\.md|\.en\.markdown', infile.name)

            if english and isEnglish:
                df = df.append(data, ignore_index=True)
            elif not english and not isEnglish:
                df = df.append(data, ignore_index=True)

    # Save for latter use
    #df.to_csv('./post_data.csv', index=False)

    return df


def preprocessor(text):
    # TODO: Remove punctuation
    # Remove frontmatter
    text = re.sub(r'^\s*---.*---\s*$', '', text,
                  flags=re.DOTALL | re.MULTILINE | re.UNICODE)
    text = re.sub(r'^\s*\+{3}.*\+{3}\s*$', '', text,
                  flags=re.DOTALL | re.MULTILINE | re.UNICODE)
    text = re.sub(r'^\s*```.*?```\s*$', '', text,
                  flags=re.DOTALL | re.MULTILINE)
    text = re.sub(r'`[^`]*`', '', text)
    text = re.sub(r'<[^>]*>', '', text, flags=re.UNICODE |
                  re.DOTALL | re.MULTILINE)
    text = text.replace('<!--more--><!--ad-->', '')
    text = re.sub(r'https?:\/\/.*[\r\n]*', '',
                  text, flags=re.MULTILINE | re.UNICODE)
    text = re.sub(r'[#|*|\[\]:.,]', '', text, flags=re.UNICODE)
    text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~]', '', text)
    text = re.sub(r'\d*', '', text)
    #text = re.sub(r'[\W]+', ' ', text.lower(), flags=re.UNICODE)

    return text


def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

# Cambiamos a este stemmer que tiene soporte para español


def tokenizer_snowball(text):
    stemmer = SnowballStemmer("spanish")
    return [stemmer.stem(word) for word in text.split()]


def stop_removal(text, stops_w):
    return [w for w in text.split() if w not in stops_w]


def generateTfIdfVectorizer(data, stop='english', max_df=0.08, min_df=8):
    tokenizer = tokenizer_snowball if stop != 'english' else tokenizer_porter

    tfidf = TfidfVectorizer(strip_accents=None,
                            max_df=max_df,
                            min_df=min_df,
                            lowercase=True,
                            stop_words=stop,
                            sublinear_tf=True,
                            tokenizer=tokenizer,
                            analyzer='word',
                            max_features=32,
                            preprocessor=preprocessor)

    X = tfidf.fit_transform(data)
    print('Features: %s' % tfidf.get_feature_names())
    return X


def KmeansWrapper(true_k, data, load=False):
    from sklearn.externals import joblib

    modelName = 'doc_cluster.%s.plk' % true_k

    if load:
        km = joblib.load(modelName)
        clusters = km.labels_.tolist()
    else:
        km = KMeans(n_clusters=true_k,
                    init='k-means++',
                    max_iter=1000,
                    n_init=10,
                    n_jobs=-1,
                    random_state=0,
                    verbose=0)
        km.fit_predict(data)
        clusters = km.labels_.tolist()
        joblib.dump(km,  modelName)

    return clusters


def elbowMethod(X, k=21):
    distortions = []
    for i in range(1, k):
        km2 = KMeans(n_clusters=i,
                     init='k-means++',
                     n_init=10,
                     max_iter=1000,
                     random_state=0,
                     n_jobs=-1,
                     verbose=0)
        km2.fit(X)
        distortions.append(km2.inertia_)
        print('k=%s, Distortion: %.2f' % (i, km2.inertia_))

    plt.plot(range(1, k), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()


def plotPCA(df, true_k):
    # Plot in 2d with PCA
    dist = 1 - cosine_similarity(X)

    MDS()
    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

    xs, ys = pos[:, 0], pos[:, 1]

    import matplotlib.cm as cm
    import numpy as np

    # set up colors per clusters using a dict
    cluster_colors = cm.rainbow(np.linspace(0, 1, true_k))

    # set up cluster names using a dict
    # cluster_names = {i: 'i' for i in range(true_k)}

    # create data frame that has the result of the MDS plus the cluster
    # numbers and titles
    df2 = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=df[0],
                            title2=df[2]))

    # group by cluster
    groups = df2.groupby('label')

    pd.set_option('display.max_rows', len(df2))
    print(df2.sort_values(by='label')[['label', 'title', 'title2']])
    df2.sort_values(by='label')[
        ['label', 'title', 'title2']].to_csv('./labels.csv')
    pd.reset_option('display.max_rows')

    # set up plot
    fig, ax = plt.subplots(figsize=(25, 25))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name'
    # lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                color=cluster_colors[name],
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(
            axis='y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')

    # ax.legend(numpoints=1)  #show legend with only 1 point

    # add label in x,y position with the label as the film title

    for i in range(len(df2)):
        ax.text(df2.ix[i]['x'], df2.ix[i]['y'], df2.ix[i]['title'], size=4)

    # plt.show() # show the plot
    plt.savefig('test.pdf', format='pdf')  # , dpi=600)
    plt.savefig('test.eps', format='eps')  # , dpi=600)
    plt.savefig('clusters_small_noaxes.png')  # , dpi=600)
    plt.close()

    class TopToolbar(mpld3.plugins.PluginBase):
        """Plugin for moving toolbar to top of figure"""

        JAVASCRIPT = """
        mpld3.register_plugin("toptoolbar", TopToolbar);
        TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
        TopToolbar.prototype.constructor = TopToolbar;
        function TopToolbar(fig, props){
            mpld3.Plugin.call(this, fig, props);
        };

        TopToolbar.prototype.draw = function(){
          // the toolbar svg doesn't exist
          // yet, so first draw it
          this.fig.toolbar.draw();

          // then change the y position to be
          // at the top of the figure
          this.fig.toolbar.toolbar.attr("x", 150);
          this.fig.toolbar.toolbar.attr("y", 400);

          // then remove the draw function,
          // so that it is not called again
          this.fig.toolbar.draw = function() {}
        }
        """

        def __init__(self):
            self.dict_ = {"type": "toptoolbar"}

    # create data frame that has the result of the MDS plus the cluster
    # numbers and titles
    df3 = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=df[0]))

    # group by cluster
    groups = df3.groupby('label')

    # define custom css to format the font and to remove the axis labeling
    css = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }

    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }

    svg.mpld3-figure {
    margin-left: 200px;}
    """

    # Plot
    fig, ax = plt.subplots(figsize=(25, 25))  # set plot size
    ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

    # iterate through groups to layer the plot
    # note that I use the cluster_name and cluster_color dicts with the 'name'
    # lookup to return the appropriate color/label
    for name, group in groups:
        points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18,
                         mec='none',
                         color=cluster_colors[name])
        ax.set_aspect('auto')
        labels = [i for i in group.title]

        # set tooltip using points, labels and the already defined 'css'
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels,
                                                 voffset=10, hoffset=10, css=css)
        # connect tooltip to fig
        mpld3.plugins.connect(fig, tooltip, TopToolbar())

        # set tick marks as blank
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        # set axis as blank
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    ax.legend(numpoints=1)  # show legend with only one dot

    mpld3.display()  # show the plot

    # uncomment the below to export to html
    html = mpld3.fig_to_html(fig)
    mpld3.save_html(fig, 'name.html')

# nltk.download('stopwords')


stop = stopwords.words('spanish')
stopE = stopwords.words('english')

stop = stop + stopE

df = readPosts('./post', english=True)

df[1] = df[1].apply(preprocessor)  # k = 21
df.to_csv('./post_data.cleanded.csv')

# X = generateTfIdfVectorizer(df[1], stop) # k = 11
X = generateTfIdfVectorizer(df[1], stop=stop, max_df=0.7, min_df=2) # k = 3

true_k = 3
clusters = KmeansWrapper(true_k, X)

#elbowMethod(X)
plotPCA(df, true_k)
