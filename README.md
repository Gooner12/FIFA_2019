# FIFA_2019

## Overview
This project involves the analysis of football players present in the EA sports database, particularly FIFA 2019 and FIFA 2021 database. In this project, we divide the analysis task into three types. They are clustering, classification and regression analysis. Furthermore, the analysis is performed using SPARK packages. 

## Project Types
This project contains three subprojects that aim to capture specific results based on the players' data. 

* <b>FIFA_position_predict:</b> This subproject deals with identifying the best playing position for a player. First, the players are clustered in groups based on their attributes. Clustering helps us know other positions that a player can play when the team runs out of options for a position. Looking at the attributes, one can tell whether a player is a forward, midfielder or defender, but clustering provides us extra information about the suitability in other positions. Based on this, we can deploy those players in different positions when required. Finally, we use cluster information to predict the most suitable position out of the three positions.

* <b>FIFA_growth_predict:</b> This subproject aims to uncover whether a player has the potential to improve and grow further in the future. Players are categorised into four categories, namely 'No Growth', 'Low Growth', 'Mid Growth' and 'High Growth'. The end result would help football clubs identify potential talents and save money and resources used for scouting talents. To achieve this, we combine datasets from 2019 and 2021 to estimate the growth of a player using machine learning algorithms. Unfortunately, due to the imbalance in data, we fail to develop an effective predictor even after applying multiple techniques to deal with the class imbalance. 

* <b>FIFA_gain_predict:</b> This subproject analyses players to determine if any profits can be made from them by selling them after keeping them in the club for two years. Also, it assists clubs in understanding beforehand the type of players they are dealing with before starting the negotiation. It predicts the possible loss or gain that could be obtained from a player. To achieve this, we wrangled the dataset after merging both the 2019 and 2021 databases and utilised machine learning algorithms to produce the desired outcomes.

## Prerequisites
* findspark
* wget
* PySpark
* matplotlib
* pandas
* NumPy
* plotly
* seaborn, plus project-specific classes and datasets.

## Highlights
This section is dedicated to listing the important learnings obtained from this project.
* Extended the SMOTE technique for over/ undersampling of datasets containing more than two classes in PySpark.
* Development of Variation Inflation Factor (VIF) detector to remove multicollinearity in PySpark.
* Data wrangling.
* Building custom estimators and transformers.

## Limitations
* This project does not deal with goalkeepers, which makes the analysis of players suitable for goalkeeping positions inconclusive.
* Identifying the potential of players would be more accurate if more data is available.

## Further Information
The notebooks in this project have been developed in Google Colab. However, the notebooks can be run in Jupyter as well given that the configurations required for setting up this project are met, such as the installation of PySpark and jdk.

<b><i>Note:</b></i> Please view the notebooks in google collab to see the results and steps in a prettier format.
