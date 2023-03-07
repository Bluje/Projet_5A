# LinK - Projet de compréhension et de visualisation des liens existants entre différentes terminologies et différents articles
## Contexte

Il s'agit d'un projet réalisé dans le cadre de la thèse d'Albane Gril doctorante de 3eme année effectue actuellement (entre janvier 2021 et décembre 2023) une thèse qui vise à disposer d’un prototype d’environnement de visualisation intégrant des tableaux de bord de suivi pédagogique et de suivi de l’étude d’impact dans le cadre du développement de compétences écrites d’étudiant.es en français académique à l’université (Sujet de thèse d’Albane Gril). Cette thèse est encadrée par Valérie Renault, Madeth May et Sébastien Georges, au sein du LIUM (Laboratoire d’Informatique du Mans) et du CREN (Centre de recherche en éducation de Nantes).

Problématique : Comment fournir aux utilisateur.trices un outil qui permette de comprendre voire de visualiser les liens entre plusieurs terminologies dans différents domaines d’application, grâce à l’intelligence artificielle?

## Résultats
Ce code permet d'obtenir une visualisation de clustering d'articles avec MatPlotlib et avec Plotly à l'aide du score de l'algorithme TF-IDF et également de la méthode Word2Vec. Ce clustering se fait dans le premier cas selon les mots-clés les plus fréquents des articles tandis que cela s'effectue selon tous les mots peut importe leur fréquence dans le second cas.

## Utilisation
Pour utiliser ce code, il faut mettre en entrée un fichier en format JSON qui contient les fields "title", "container-title", "abstract" (au moins d'un de ces fields).
