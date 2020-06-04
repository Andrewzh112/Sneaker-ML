# Sneaker ML [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Andrewzh112/Sneaker-ML/issues)[![HitCount](http://hits.dwyl.com/Andrewzh112/Sneaker-ML.svg)](http://hits.dwyl.com/Andrewzh112/Sneaker-ML)
This project is aimed to discover the differences in hyped sneakers and regular shoes as well as explore differences _hyped_ sneaker designs. _Hyped_ sneakers are defined as sneakers that are listed on popular resale sites such as **Goat**, **Flight Club** and **StockX**. The resale market of these coveted sneakers has flourished in the last couple of years. While some people still view _hyped_ sneakers has a past time hobby, others actually became professional resellers. Additionally, sneaker companies also interested to find out what “works” and what are some design practices that can make their products excel in resell markets.

Since _hyped_ sneakers have great business value for sneaker resellers and shoe brands alike, it is then appropriate to try to identify design characteristics in these _hyped_ sneakers that set them apart. The purpose of the project is to explore _hyped_ sneaker designs, what differs them from “everyday” shoes, as well as what are some character defining designs that different brands are doing to distinguish from their counter parts. 

#### Motivation
This project is aimed to use various machine learning techniques to look at the _style_ of a _hyped_ sneaker. Different techniques like image clustering, classification and generative adversarial networks are some of the ways the project used to tackle this problem. 

#### Project Procedure

1. Gather shoe images from **Goat**, **Flight Club**, **StockX** and **Amazon**.
2. Preprocess image data by first manually removing images that do not belong to the dataset at all (not shoes) or shoes that are oriented very differently from the rest (shoes pointing upwards or images showing a pair of shoes instead of a single shoe). Next, we can pad all the images that are not of the same size to the same size to make the subsequent training jobs easier.
3. Perform exploratory data analysis on the dataset.
4. Perform clustering and make eigen images of the clusters to identify key cluster defining factors.
5. Binary and multiclass classification and use of Grad-CAM to explore learning algorithm's decisions.
6. GANs image generation.

![](misc/GAN.gif)