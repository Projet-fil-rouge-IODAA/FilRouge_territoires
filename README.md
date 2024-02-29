# Contexte et objectif du projet 
Les données satellites représentent un outil majeur dans l'étude des territoires, cela de par la richesse en informations que celles-ci apportent de manière régulière, et de par la précision des informations collectées. La mise en évidence de motifs sur des données spatiales est un domaine qui est très présent dans la littérature, mais ceci majoritairement dans le cadre d'études géomatiques à un instant donné, donc sans la prise en compte de l'évolution.\\

Dans le cadre de ce projet, nous souhaitons étudier les évolutions du territoire, et donc les motifs apparents entre séries temporelles en prenant compte de la dimension spatiale.
La zone considérée est une zone d'environ 200 km\textsuperscript{2} entre L'Essonne, les Hauts-de-Seine et le Val-de-Marne entre l'aéroport Paris-Orly et la commune des Ulis, et  comprenant le cluster Paris-Saclay (et le campus Agro Paris-Saclay d'AgroParisTech).\\

Les données à notre disposition comptent 177 images satellitaires issues de Sentinel-2A et Sentinel-2B, dont 141 sont déjà classées comme ne comptant pas de couvert nuageux lors d'un précédent prétraitement. Ces images présentent 4 bandes : proche infrarouge (NIR), rouge (R), vert (G) et bleu (B). Le poids de chaque image est de 17 Mo et l'étendue de ces images est de 973 x 2182 pixels, avec une résolution de 10 m, pour une couverture temporelle étendue entre décembre 2015 et octobre 2023.\\

# Transformation des données brutes et création des jeux de données

# Exécution de Time2Features

# Exécution de CollaborativeClustering

# Format des résultats

# Structure du dépôt:
```
.
├── DL
├── data
│   ├── processed
│   └── raw
├── docs
├── models
├── notebooks
│   ├── __pycache__
│   └── graphic_src
├── references
├── reports
└── scr
    ├── data
    ├── features
    ├── models
    └── visualization
```

# Bibliographie

Adwan, S. et Arof, H. (2010). A novel double stage dynamic time warping algorithm for
image template matching. In Proceeding of the 6th IMT-GT Conference on Mathematics,
Statistics and its Applications (ICMSA 2010), University Tunku Abdul Rahman, Kaula
Lumpur Malaysia, pages 667–676.
Bocher, E. et Petit, G. (2013). Mutation des territoires, application à l’étude des limites
communales. Les Journées de la géomatique des pays de la Loire.
Bonifati, A., Buono, F. D., Guerra, F. et Tiano, D. (2022). Time2feat: learning interpretable
representations for multivariate time series clustering. Proc. VLDB Endow., 16(2):193–201.
Christ, M., Braun, N., Neuffer, J. et Kempa-Liehr, A. W. (2018). Time series
feature extraction on basis of scalable hypothesis tests (tsfresh – a python package).
Neurocomputing, 307:72–77.
Furtună, T. F. (2008). Dynamic programming algorithms in speech recognition. Revista
Informatica Economică nr, 2(46):94.
Hall-Beyer, M. (2017). Practical guidelines for choosing glcm textures to use in landscape
classification tasks over a range of moderate spatial scales. International Journal of Remote
Sensing, 38(5):1312–1338.
Haralick, R. M. (1979). Statistical and structural approaches to texture. Proceedings of the
IEEE, 67(5):786–804.
Li, H. (2019). Multivariate time series clustering based on common principal component analysis.
Neurocomputing, 349:239–247.
Li, H., Lin, C., Wan, X. et Li, Z. (2019). Feature representation and similarity measure based
on covariance sequence for multivariate time series. IEEE Access, 7:67018–67026.
Luqian, S. et Yuyuan, Z. (2021). Human activity recognition using time series pattern
recognition model-based on tsfresh features. In 2021 International Wireless Communications
and Mobile Computing (IWCMC), pages 1035–1040. IEEE.
Siebert, J., Groß, J. et Schroth, C. (2021). A systematic review of packages for time series
analysis. Engineering Proceedings, 5(1):22.
S.Khedairia, M. T. (2022). A multiple clustering combination approach based on iterative
voting process. Journal of King Saud University - Computer and Information Sciences,
34:1370–1380.
Tiozzo, A. (2022). Dynamic time warping for time series classification. Medium.
Veltz, P. (2018). Introduction - regions and territories: Evolutions and changes. Economie et
Statistique / Economics and Statistics, page 5–17.
Wang, G.-J., Xie, C., Han, F. et Sun, B. (2012). Similarity measure and topology evolution
of foreign exchange markets using dynamic time warping method: Evidence from minimal
spanning tree. Physica A: Statistical Mechanics and its Applications, 391(16):4136–4146.
Yadav, M. et Alam, M. A. (2018). Dynamic time warping (dtw) algorithm in speech: a review.
International Journal of Research in Electronics and Computer Engineering, 6(1):524–528
Zhang, Y. (1999). Optimisation of building detection in satellite images by combining
multispectral classification and texture filtering. ISPRS journal of photogrammetry and
remote sensing, 54(1):50–60.

# Liens utiles : 
Liste de taches : [Taches](https://docs.google.com/spreadsheets/d/12IO9i0rIVR-RKDQXc6y8nDWCuq3UjpeH08N_X2qjHiY/edit#gid=0)
Diaporama final : [Diaporama](https://docs.google.com/presentation/d/18Yu9UxA4SBvoR4pk4BbsPQTedtHY0RUU/edit?usp=sharing&ouid=105910814065404947173&rtpof=true&sd=true)
