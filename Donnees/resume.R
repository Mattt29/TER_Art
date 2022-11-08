setwd("C:/Users/matth/Desktop/Cours/Paul Va/MIASHS/Master/S1/projet/TER_Art/Donnees") #Ouvrir l'emplacement du fichier
library(readr)
artiste <- read_csv("artiste.csv", col_names = FALSE)
summary(artiste)
boxplot(artiste[,2])
