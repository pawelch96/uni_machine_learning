import pandas as pd

BASE_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/"
datasetsdict = {
"Dermatology" : BASE_URL + "dermatology/dermatology.data",
"Soybean" : BASE_URL + "soybean/soybean-large.data",
    # "Automobile" : BASE_URL + "autos/imports-85.data", strings odpada
    # "Horse Colic" : BASE_URL + "horse-colic/horse-colic.data", lots of ()???)
    # "Parkinsons" : BASE_URL + "parkinsons/parkinsons.data", ex 'phon_R01_S49_6,114.56300,119.16700,86.64700,0.00327,0.00003,...
    # "Flags" : BASE_URL + "flags/flag.data", 1 col - nameof country, last - string with color, jeden row dla jednej klasy
    # "Ionosphere" : BASE_URL + "ionosphere/ionosphere.data", positive and negative, but probably ok
    # "Cervical cancer" : BASE_URL + "00383/risk_factors_cervical_cancer.csv", ? odpada fchuj, same braki
    "Water treatment" : BASE_URL + "water-treatment/water-treatment.data",
    "Cylinder Bands" : BASE_URL + "cylinder-bands/bands.data",
    "Abscisic acid" : BASE_URL + "abscisic-acid/plantCellSignaling.data",
    "Heart Disease" : BASE_URL + "heart-disease/new.data",
    "Mushroom" : BASE_URL + "mushroom/agaricus-lepiota.data",
    "Image segmentation" : BASE_URL + "image/segmentation.data",
    "Relation Network" : BASE_URL + "00220/Relation%20Network%20(Directed).data",
    "Lymphography" : BASE_URL + "lymphography/lymphography.data",
    "Meta-data" : BASE_URL + "meta-data/meta.data"
}
