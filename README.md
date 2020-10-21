# Projet 5 : Segmenter des clients d'un site e-commerce.
*Pierre-Eloi Ragetly*

Ce projet fait parti du parcours DataScientist d'Open Classrooms.

L'objectif principal est de réaliser une segmentation des clients d'un site e-commerce sur des données anonymes (notamment pas de données sur le sexe ou l'âge).  
Les données utilisées proviennent du projet kaggle [*Brazilian E-Commerce*](https://www.kaggle.com/olistbr/brazilian-ecommerce).

L'analyse a été découpée en deux notebooks:
- **Project5_Data_Wrangling** qui regroupe l'analyse exploratoire des données
- **Project5_Clustering** qui regroupe les différents modèles de clustering testés

*Une version html des notebooks a été ajoutée de manière à pouvoir les visualiser sans avoir à installer les librairies requises*

Les librairies python nécessaires pour pouvoir lancer le notebook sont regroupées dans le fichier *requirements.txt*  
Toutes les fonctions créées afin de préparer les données (incluant un pipeline à la fin) ont été regroupées dans le fichier **functions/wrangling.py**.  
Les figures ont été regroupées dans le dossier **charts**.

Durant l'étude, les principaux algorithmes de clustering ont été testées :
- Agglomeratif Clustering
- K-Means
- DBSCAN
- Gaussian Mixture
