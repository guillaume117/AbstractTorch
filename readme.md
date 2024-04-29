### AbstractTorch
Ceci est un prototype de moteur d'évaluation s'appuyant sur Torch. L'idée est de surcharger nn.Linear avec des méthodes hybrides, gérant un flux abstrait pour l'évaluation et un flux concret. La simultanéité des deux flux permet d'oberver la position du centre du zonotope par rapport à la valeur réelle de la sortie de la fonction. 

L'implémentation du modèle abstrait comporte une quantité fixe de symboles qui sont gérés comme des épaisseurs de batch. La dernière épaisseur de batch correspond au symbole poubelle, les opération linéaires sont opérées pour cette épaisseur par la valeur absolu de la matrice des poids. 

L'option add_symbol permet de générer de nouveaux symboles/ 
L'inférence peut se faire soit sur cpu (device=torch.device('cpu')) ou carte graphique (device = torch.device('cuda'ou 'mps'))

Pour l'instant sont implémentées les classes conv2D, Linear , et ReLU. 
        #TODO implémenter maxpool2D, ... 
    

On tire profit de la structure de base
de la méthode forward. Au lieu de considérer un batch, on considère une entrée en dimension 0 avec dans les dimensions habituelles du batch des couches de symbole. Une couche (un épaisseur de batch) représente
un symbole abstrait. La dernière couche correspond au symbole poubelle. 


La couche 0 représente le centre du zonotope
Les couches suivantes représentent les symboles. Elles sont calculées pour les opération linéaires (Linear et Conv2D) par 
$$\textbf{W}(x_\epsilon)+\textbf{b}-(\textbf{W}(0)+\textbf{b})$$
    x[1:]=lin(x_epsilon)-lin(torch.zeros_like(x_epsilon))

La derniere couche est toujours celle du bruit poubelle. Les opérations linéaires  sont  calculés de la façon suivantes: 

$$\textbf{|W|}(x_\epsilon)+\textbf{b}-(\textbf{|W|}(0)+\textbf{b})$$


Pour implémenter le tenseur linéaire représentant la valeur absolue, on duplique la couche lin ou conv et on applique la valeur absolue à la matrice de poids. 


Cette dernière couche peut être nulle si les symboles générés sont projetés sur une nouvelle dimension. 



