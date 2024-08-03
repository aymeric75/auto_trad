## fcts implementees

# return_train_from_liste
entree: liste de types (e.g. ha, rsi etc), contraintes de tf, temps, heure etc

sortie1: liste d'array np, où chaque array a shape (nbre examples, nbre_time_steps)

sortie2: liste des noms <=> à chaque array



# from_list_of_datas_and_names_save_train_data
entree: la sortie de la fct ci-dessus

sortie: aucune, mais save l'ensemble des arrays donnees en entrée
en tant qu'un seul array de dimension (nbre_examples, nbre_time_steps, nb_arrays)