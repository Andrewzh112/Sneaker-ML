import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import decomposition
import pandas as pd
import numpy as np


def plot_RBG_dist(rbg,title,x_range):
    """"Plot Red Green Blue Density Graphs"""
    
    plt.figure(figsize=(12,10))
    color = ['red','green','blue']
    for i,col in enumerate(color):
        sns.distplot(rbg[i],color=col)
        plt.ylim(0,0.05)
        plt.xlim(x_range)
    plt.tick_params(labelleft='off', 
                    labelbottom='off', 
                    bottom='off',
                    top='off',
                    right='off',
                    left='off', 
                    which='both')
    plt.title(f'RBG distribution for {title} Shoes')
    
    
def average_shoe(img_array,n_components=500,shape=(224, 224, 3)):
    """Plot the 'average' shoe"""
    
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(img_array)
    plt.tick_params(labelleft='off', 
                    labelbottom='off', 
                    bottom='off',
                    top='off',
                    right='off',
                    left='off', 
                    which='both')
    plt.imshow(pca.mean_.reshape(shape).astype(np.uint8))
    
def plot_tsne(tsne_results, freevar, hue='hyped', classes=2, title='Hyped Shoes and Non-Hyped Shoes'):    
    tsne_aker_df = pd.DataFrame()
    tsne_aker_df['tsne-one'] = tsne_results[:,0]
    tsne_aker_df['tsne-two'] = tsne_results[:,1]
    tsne_aker_df[hue] = 0
    
    if hue == 'hyped':
        for i in range(tsne_aker_df.shape[0]):
            tsne_aker_df[hue].iloc[i] = i > freevar - 1
    else:
        tsne_aker_df[hue] = freevar
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x='tsne-one', y='tsne-two',
        hue=hue,
        palette=sns.color_palette('hls', classes),
        data=tsne_aker_df,
        legend='full',
        alpha=0.3
    )
    plt.title(title)