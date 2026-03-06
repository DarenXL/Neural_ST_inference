#import "@preview/diatypst:0.8.0": *
#import "@preview/cetz:0.4.2"
#import "@preview/touying:0.6.2": *

//typst watch Slides/slides-stage/main.typ --open --font-path Slides/slides-stage/Inconsolata/

#set text(font: "Inconsolata")
#set text(lang: "fr")

// #let theme_color = rgb("#FFC662")
#let theme_color = rgb("#7AA7F1")

#show: slides.with(
  title: "Introduction stage", // Required
  subtitle: "Estimation de modèles spatio-temporels par Réseaux de Neurones sur Graphes",
  date: "9 mars 2026",
  authors: ("Thomas Romary, Nicolas Dessassis, Alexandre Loret et Lucia Clarotto"),

  ratio: 16/9,
  layout: "medium",
  title-color: theme_color, 
  toc: true,
  theme: "normal",
  count:"number",
  footer-subtitle:"Estimation de modèles spatio-temporels par GNN"
)

= Contexte

== Informations générales

La thèse a lieu entre le laboratoire de géoscience de *Mines Paris - PSL* et l'*AgroParisTech*.

- Encadrants Mines Paris : Thomas Romary et Nicolas Dessassis
- Encadrante AgroParisTech : Lucia Clarotto

#v(10pt)
La thèse a pour objectif de trouver une *méthode* efficace pour *estimer* le *modèle géostatistique* d'un *phénomène spatio-temporels*.

La géostatistique est lien entre la *géoscience* et les *statistiques*.
- *Géoscience :* Étude des sciences de la Terre comme la géologie, l'océanographie, etc.
- *Statistiques :* Étude d'un phénomène à travers des observations.

== Phénomène spatio-temporel

Un *phénomène spatio-temporel* peut par exemple être la pollution dans l'air. Dans ce cas, les observations sont :
- Réparties de manière irrégulière dans l'espace (des points sur un carte)
- Discontinues dans le temps (par exemple toutes les heures)

#v(10pt)
Ainsi, un *modèle* de ce phénomène permettrait de connaître le taux de pollution n'importe où, n'importe quand.

#grid(
  columns: (50%,50%),
  //mage("example_mask.pdf", height: 50%),
  image("fig/example_animated.gif",format:"gif", height: 50%),
  align(horizon)[note : les observations (2ème ligne) ont été représenté par des pixels mais ce sont des points]
)


== Modèle spatio-temporel

*Données* #linebreak() Observations bruitées $y (bold(s)_i, t_j)$ à certains points dans l'espace $(bold(s)_1, ..., bold(s)_n)$ de $cal(D) subset RR^d$ et pas de temps $(t_1, ..., t_m)$ de $cal(T) subset RR$.
      
*Modèle* #linebreak() Pour chaque $i in {1, ..., n}$ et $j in {1, ..., m}$, on suppose que les observations $y(bold(s)_i,t_j)$ sont la réalisation d'une variable aléatoire $Y(bold(s)_i,t_j)$

#align(center)[
    #box(stroke: (paint:theme_color, thickness:2pt), 
    inset:(top:30%, left:10%, right:10%, bottom:30%))[
      $ Y(bold(s_i),t_j) = mu(bold(s_i),t_j) + Z(bold(s_i), t_j) + tau epsilon_(i,j), quad tau > 0 $
    ]
]

      - $mu$ l'*effet fixe*
      - $Z$ un champs gaussien (zero-mean) aléatoire stationnaire de *fonction de covariance* $ C (bold(h), u) = "Cov"(Y(bold(s)+bold(h),t+u), Y(bold(s),t)) $ 
      - $epsilon_(i,j) ~ cal(N)(0,1)$ iid du *bruit*



== Détails Champs Gaussien Aléatoire

*Écriture matricielle*
$ bold(Z) = mat(
  bold(Z)_t_1;
  bold(Z)_t_2;
  ...;
  bold(Z)_t_m;
) "avec" bold(Z)_t_j = mat(
  Z(bold(s_1), t_j);
  Z(bold(s_2), t_j);
  ...;
  Z(bold(s_n), t_j);
) $
$ bold(Z) ~ cal(N)(bold(0),bold(Sigma)) "avec" 
bold(Sigma) = mat(
  dots.down, dots.c, dots.up;
  dots.c, "Cov"(Z(bold(s)_i,t_j),Z(bold(s)_i',t_j')), dots.c;
  dots.up, dots.c, dots.down
) $
la même $"Cov"$ que la fonction de covariance $C(bold(h),u)$.
#line(length:100%, stroke: theme_color)
*Définitions :*
- Stationnaire : Dépend uniquement de la distance entre les points
- Isotrope : $C(bold(h),u) = C(-bold(h),u) = C(bold(h),-u) = C(-bold(h),-u)$

== Fonction de covariance paramétrisée

On se place dans un contexte paramétrisé sans effet-fixe :
$ Y(bold(s_i),t_j; theta) = Z(bold(s_i), t_j; theta) + tau epsilon_(i,j), quad tau > 0 $
On note les observations $bold(z)_"obs"$ et les suppose être une réalisation de la variable aléatoire ci-dessus. Cela implique de choisir une *fonction de covariance $C_theta$* (et donc une matrice covariance $bold(Sigma)_theta$).

#v(10pt)
L'objectif est d'*estimer les paramètres* de la covariance qui a généré les observations.
#linebreak()

#line(length:100%, stroke: theme_color)

L'Estimation du Maximum de vraisemblance (MLE) est une méthode classique. Mais la méthode devient très coûteuse lorsque que le nombre de données augmente.
$ cal(L)(theta | bold(y)) = -(n m)/2 log(2 pi) - 1/2 log det(Sigma_theta) - 1/2 norm(bold(y) - mu id)^2_(Sigma^(-1)_theta) $

== Exemple de modèle
Un champs gaussien aléatoire avec covariance de Matérn 3D (2D spatial + 1D temporel). 
$ C_theta (bold(h), u) = sigma^2 2^(1-nu)/(Gamma(nu)) (sqrt(2nu) sqrt((h_1 / rho_s)^2 + (h_2 / rho_s)^2  + (u / rho_t)^2))^nu K_nu (sqrt(2nu) sqrt((h_1 / rho_s)^2 + (h_2 / rho_s)^2  + (u / rho_t)^2)) $ avec $Gamma$ la fonction Gamma et $K_nu$ la fonction de Bessel modifée du second ordre. $nu$ est fixé à 1.
Une *advection* est ajoutée à la main par dessus. #linebreak()
Cela donne 4 paramètres à estimer : $"adv"$, $rho_s$, $rho_t$ et $sigma$
#grid(
  columns: (75%,25%),
  image("fig/example.pdf", height: 45%),
  align(horizon)[note : cas 1D saptial + 1D temporel #linebreak() note : isotropie si $"adv = 0"$]
)

== Simulation-Based Inference (SBI)

#cetz.canvas(length: 2.5cm, {
  import cetz.draw: *

  let l = .65
  set-style(stroke: (thickness: 1.5pt))

  rect((0,0), (4*l, 2.25*l), name: "step1", stroke: (paint: theme_color))
  content("step1", align(center + horizon, text()[
    *1. Génération du dataset $(bold(z), theta)$* #linebreak()
    #box(width: l*260pt)[
$theta^((1)), ... theta^((N))$ sont tirés d'une loi a priori choisie $p(theta)$. #linebreak()
$bold(z)^((1)), ..., bold(z)^((N))$ sont simulées à partir du modèle choisi $Z(bold(s),t ;theta)$.]]))

  line("step1.east", stroke: 1pt, (5*l, l*2.25/2), mark: (end: "straight", length:5pt, width:10pt))

  rect((5*l, 0), (9*l, 2.25*l), name: "step2", stroke: (paint: theme_color))
  content("step2", align(center + horizon, text()[*2. Entraînement de $hat(theta)_gamma (dot)$* #linebreak() 
  // $hat(theta)_gamma :  cal(D) times cal(T) |-> RR^p$ #linebreak() 
  #box()[$ gamma^* = arg min_gamma sum_(l=1)^N L{theta^((l)), hat(theta)_gamma (bold(z)^((l)))} $] #linebreak()  #box(width: l*260pt)[On entraîne le réseau de neurones avec la perte $L$.]]))

  line("step2.south", stroke: 1pt, (2*l, -l/2), mark: (end: "straight", length:5pt, width:10pt))

  rect((0, -l/2), (4*l, -2.5*l), name: "step3", stroke: (paint: theme_color))
  content("step3", align(center + horizon, text()[*3. Estimation* #linebreak()  $hat(theta)_"obs" = hat(theta)_(gamma^*)(bold(z)_"obs")$ #linebreak() #box(width: l*260pt)[Estimer les paramètres d'une observation de demande qu'une forward pass. SBI est dite "amortie".]]))

  line("step3.east", stroke: 1pt,  (5*l, -1.5*l), mark: (end: "straight", length:5pt, width:10pt))

  rect((5*l, -l/2), (9*l, -2.5*l), name: "step4", stroke: (paint: theme_color))
  content("step4", align(center + horizon, text()[*4. Prediction* #linebreak() 
  #box(width: l*260pt)[La fonction de covariance estimée $C_(hat(theta)_"obs")$ nous permet de prédire le champs $Z$ à des localisations non-observées #linebreak() $(bold(s)', t') in cal(D) times cal(T)$ par krigeage.]]))
})

== Prédiction

Krigeage
- utilisé pour estimer les valeurs à des localisations non mesurées $(s'_i,t'_j)$ en fonction de valeurs connues aux localisations $(s_i,t_j)$ à proximité
- On suppose $z_"obs" (s_i,t_j)$ réalisation de $ Z(bold(s_i), t_j; theta^*) + tau epsilon_(i,j) $
- On veut prédire $z_"pred" (s'_i,t'_j)$ "réalisation" de $ Z(bold(s'_i), t'_j; theta^*) $
- Au format matriciel, grâce à $C_(theta^*)$, on construit :
    $ (bold(Z), bold(Z)') ~ cal(N)(0, mat(bold(V)_"obs-obs",bold(Sigma)_"obs-pred";bold(Sigma)_"pred-obs",bold(Sigma)_"pred-pred")
    " avec " bold(V)_"obs-obs" = bold(Sigma)_"obs-obs" + tau^2"id" $
- Espérance conditionnelle :
    $ bold(z)_"pred" = EE[bold(Z)' | bold(Z)=bold(z)_"obs"] = bold(Sigma)_"pred-obs" bold(V)_"obs-obs"^(-1)(bold(z)_"obs") $

#pagebreak()

Nouvelle simulation non-conditionelle
- On peut aussi faire une simulation avec les paramètres trouvés
- Nos observations ne seront pas ces nouvelles simulations

#v(15pt)

Simulation conditionelle
- On commence par un $"krigeage"$ comme précédemment
- On fait une $"simulation non-conditionelle"$ sur une grille complète
- On extrait les valeurs de cette simulation à l'endroit de nos observations
- On fait un $"krigeage non-conditionel"$ grâce aux valeurs extraites
- La $"simulation conditionelle"$ est $ "krigeage" + "simulation non-conditionelle" - "krigeage non-conditionel" $

== Détails des données dans la thèse

*1D spatial $times$ 1D temporel :*
- Plus de clarté sur des slides/poster
- Plus rapide à simuler/entraîner

#v(15pt)
*Génération du dataset :*
- Tirage des $theta^((1)), ... theta^((N))$ selon une loi uniforme
- Pour chaque $theta$, on simule une *grille entière* (de taille $30 times 30$, même décalage)
- A chaque époque de l’entraînement
  - On génère un masque de même taille et on met les valeurs sous le masque à 0
  - Une observation prise en entrée par le réseau est de la forme (données, masque)

== Choix Réseau de neurones dans la thèse

2D CNN : *Conv2D $->$ Conv2D $->$ MaxPooling*

*Architecture deepset :* #linebreak()
// Les données répliquées peuvent (générées sur un même $theta$) peuvent mener à du biais.
// Un batch peut contenir plusieurs répliques donc on n'utilise l'architecture *deepset*.
Si on a des données répliquées (on les sait générées sur un même $theta$), on veut les utliser pour améliorer l'estimation du modèle.
On passe par une architecture *deepset*.

#cetz.canvas(length: 3cm, {
  import cetz.draw: *

  // circle((0,0), radius: 1)

  // rect((-1, -1), (1, 1))

  set-style(stroke: (paint: black, thickness: 2pt))

  let l = .5
  let r = l*2


  rect((0,0),(l*3,l), name:"data")
  content("data.north", [\ \ *Data*])
  content("data", [\ Images + Masque])

  line((3.5*l, l/2), (5*l, l/2), mark: (end: "stealth"), name:"nn")
  content(("nn.start", 50%, "nn.end"), anchor: "south", [*Neural network* \ \ ])
  content(("nn.start", 50%, "nn.end"), anchor: "north",  align(center)[#v(l*16pt) 2D CNN])

  rect((5.5*l,0),(7*l,l), name:"data")
  content("data", [*Résumé*])

  line((7.5*l, l/2), (9*l, l/2), mark: (end: "stealth"), name:"nn")
  content("nn",  anchor: "south", [*Neural network* \ \ ])
  content("nn", anchor: "north", [\ Couche Dense])

  rect((9.5*l,0),(11*l,l), name:"data")
  content("data", [*$theta$*])

  content((l*1.35,-l/2),(l*1.35,-l/2),text(theme_color, size:l*30pt)[*$dots.c$*])


  set-style(stroke: (paint: theme_color, thickness: 2pt))
  
  rect((0,0-r),(l*3,-r+l), name:"data")
  content("data.north", text(theme_color)[\ \ *Data Réplique*])
  content("data", text(theme_color)[\ Images + Masque])

  line((3.5*l, -r+l/2), (5*l, -r+l/2), mark: (end: "stealth"), name:"nn")
  content(("nn.start", 50%, "nn.end"), anchor: "south", [*Neural network* \ \ ])
  content(("nn.start", 50%, "nn.end"), anchor: "north",  align(center)[#v(l*16pt) 2D CNN])

  rect((5.5*l,-r),(7*l,-r+l), name:"data")
  content("data", text(theme_color)[*Résumé*])

  line((6.25*l, -r+1.25*l), (6.25*l, -.25*l), mark: (end: "stealth"), name:"nn")
  content("nn",  anchor: "west", text(theme_color)[ #h(l*10pt) *Moyenne*])
})

== Résultats

#align(center)[
#image("fig/visualisation - jointly max 1D-Space CNN [0].pdf")
#image("fig/score_test_alt - jointly max 1D-Space CNN.pdf")
]

== BONUS : NLS

Il a été montré que la vraisemblance peut être approchée de la façon suivante :
$ L(theta|y) = frac(h(y,theta),1-h(y,theta)) $
avec $h(y,theta)$ un classifier qui associe $C=1$ si la réalisation $y$ et le paramètre $theta$ sont dépendant, $C=2$ si indépendants.

#cetz.canvas(length: 3cm, {
  import cetz.draw: *

  // circle((0,0), radius: 1)

  // rect((-1, -1), (1, 1))

  set-style(stroke: (paint: black, thickness: 2pt))

  let l = .5
  let r = l*3


  rect((0,0),(l*3,l), name:"data")
  content("data.north", [\ \ *Data*])
  content("data", [\ Graphes ou Images])

  line((3.5*l, l/2), (5*l, l/2), mark: (end: "stealth"), name:"nn")
  content(("nn.start", 50%, "nn.end"), anchor: "south", [*Neural network* \ \ ])
  content(("nn.start", 50%, "nn.end"), anchor: "north",  align(center)[#v(l*16pt) 2D CNN])

  rect((5.5*l,0),(7*l,-l*0.5), name:"nls", stroke: (paint: theme_color))
  content("nls", text(theme_color)[*$theta$*])
  rect((5.5*l,0),(7*l,l), name:"data")
  content("data", [*Résumé*])

  line((7.5*l, l/2), (9*l, l/2), mark: (end: "stealth"), name:"nn")
  content("nn",  anchor: "south", [*Neural network* \ \ ])
  content("nn", anchor: "north", [\ Couche Dense])


  rect((9.5*l,0),(11*l,l), name:"data")
  content("data", text(theme_color)[*1 ou 2*])
})

Ce classifier permet de trouver la vraisemblance pour un couple (donnée, $theta$). Pour une donnée, le meilleur $theta$ est trouvé par gridsearch ou optimisation.

== Structure du code

- *Création de Variables*
`data_path = create_data_path("./code/data/spatiotemporal_adv")
res_path = create_res_path("./code/results/spatiotemporal")

batch_size = 50
nb_epochs = 50
freq_epochs_test = 10

time_size = 5
space_dim = 2
space_size = 30

deepset_size = 5
missing_ratios = [.90,.75]

train_size = 5
test_size = 5`

- *Création du modèle*
`model = AdvectionModel()

dataset_train = GenericSimulationDataset(
    model=model,
    theta_size=train_size,
    time_size=time_size,
    space_dim=space_dim,
    space_size=space_size,
    deepset_size=deepset_size,
)
dataset_train.generate(data_path / "train")

loader_train = DataLoader(
  dataset_train, batch_size=batch_size, shuffle=True, collate_fn=deepset_collate
)
`

$->$ idem dataset test

#pagebreak()

- *Entraînement*
`CNN = CNN_ST_network(
    in_channels=2,
    out_dim=dataset_train.model.nb_params,
    spatial_dim=space_dim,
    pool_option="separately",
).to(device)
trainer_separately = Trainer(CNN, device, res_path)
trainer_separately.train(
    train_loader=loader_train,
    test_loader=loaders_test[0],
    nb_epochs=nb_epochs,
    freq_epochs_test=freq_epochs_test,
    print_result=True,
)
for loader in loaders_test:
    trainer_separately.test(loader)`

- *Visualisation des résultats*
`trainers = [trainer_separately] 

train_graph(trainers, res_path, num_dataset=0)
time_hist(trainers, res_path, num_dataset=0)
test_alt_graph(trainers, res_path)

for i in range(len(datasets_test)):
    test_graph(trainers, res_path, num_dataset=i)
    target_graph(trainers, res_path, k=0, num_dataset=i)
    visualise(trainers, res_path, num_dataset=i)`

= Stage

== Adaptation au GNN

Les GNN avaient déjà été essayés sur un cas spatial uniquement. Les CNN s'étaient montrés plus efficaces mais nous voulons redonner une chance aux graphes.

*Objectif:* Adapter le code au GNN grâce à `PyTorch Geometric`.

#cetz.canvas(length: 3cm, {
  import cetz.draw: *

  set-style(stroke: (paint: black, thickness: 2pt))

  let l = .8

  rect((0,0),(l*2,l), name:"data")
  content("data.north", [\ \ *1. Data*])
  content("data", [\ #align(center)[Transformer les images \ en graphes]])

  line((2.25*l, l/2), (2.5*l, l/2), mark: (end: "stealth"), name:"nn")

  rect((2.75*l,0),(4.5*l,l), name:"data")
  content("data.north", [\ \ *2. GNN*])
  content("data", [\ #align(center)[Trouver une \ architecture]])

  line((4.75*l, l/2), (5*l, l/2), mark: (end: "stealth"), name:"nn")

  rect((5.25*l,0),(7*l,l), name:"data")
  content("data.north", [\ \ *3. Visualisation*])
  content("data", [\ #align(center)[Visualisation \ des résultats]])
})

#align(center,
image("fig/score_test_cropped.png")
)

== Data

Créer un nouvelle classe `DatasetGNN` qui héritera de `GenericSimulationDataset`.
- Méthodes en commun : `generate()`, `get_theta_tensor()`, ...
- Attributs en plus : `nb_neightbors`, ...
- Méthodes à surcharger : `show()`, `__getitem__()`, ...

#pause

#line(length:100%, stroke: theme_color)

En particulier `__getitem__(self, idx)` :
- Côté CNN (avec `PyTorch`) : 
  - Charge la grille complète puis applique et ajoute un masque
- Côté GNN (avec `PyTorch Geometric`) : 
  - Charge la même grille $->$ Applique un masque $->$ Transformation en graphe

#pause

#line(length:100%, stroke: theme_color)

A quoi ressemble un graphe en spatio-temporel ? 
- J'ai une idée
// Idée :
// - Les poids des nœuds sont les mesures
// - Les poids des arêtes sont (dist_spatiale, 0) ou (0, pas_de_temps)

#align(center,image("fig/example_animated.gif"))



== GNN


Commencer par un architecture simple pour vérifier que le code fonctionne.
- Par exemple GCN

$->$ Les fonctions d’entraînement/test devraient fonctionner sans modifications.

#pause

#line(length:100%, stroke: theme_color)
Puis réfléchir à une autre architecture :
- Sainsbury-Dale et al. 2025
- Autre

== Visualisation

Visualiser et comparer les résultats.

