### AbstractTorch
Ceci est un prototype de moteur d'évaluation s'appuyant sur Torch. L'idée est de surcharger nn.Linear avec des méthodes hybrides, gérant un flux abstrait pour l'évaluation et un flux concret. La simultanéité des deux flux permet d'oberver la position du centre du zonotope par rapport à la valeur réelle de la sortie de la fonction. 
Ce moteur est moins précis que Saimple car il ne génère pas de nouveaux symboles mais additionne les approximations dans un symbole poubelle. 

L'implémentation du modèle abstrait comporte une quantité fixe de symboles qui sont gérés comme des épaisseurs de batch. La dernière épaisseur de batch correspond au symbole poubelle, les opération linéaires sont opérées pour cette épaisseur par la valeur absolu de la matrice des poids. 

Pour l'instant sont implémentées les classes conv2D, Linear , et ReLU. 
        #TODO implémenter maxpool2D, ... 
    

L'idée est de tirer profit de la classe nn.Module de Torch en la surchargeant avec des méthodes mixes (flux concret et abstrait). On tire profit de la structure de base
de la méthode forward. Au lieu de considérer un batch, on considère une entrée en dimension 0 avec dans les dimensions habituelles du batch des couches de symbole. Une couche (un épaisseur de batch) représente
un symbole abstrait. La dernière couche correspond au symbole poubelle. 


La couche 0 représente le centre du zonotope
Les couches suivantes représentent les symboles. Elles sont calculées pour les opération linéaires (Linear et Conv2D) par 
$$\textbf{W}(x_\epsilon)+\textbf{b}-(\textbf{W}(0)+\textbf{b})$$
    x[1:]=lin(x_epsilon)-lin(torch.zeros_like(x_epsilon))

La derniere couche (bruit poubelle) est calculée par

$$\textbf{|W|}(x_\epsilon)+\textbf{b}-(\textbf{|W|}(0)+\textbf{b})$$


Pour implémenter le tenseur linéaire représentant la valeur absolue, on duplique la couche lin ou conv et on applique la valeur absolue à la matrice de poids. 

