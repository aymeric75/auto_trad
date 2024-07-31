

# *_date is ddmmyy
# *_h is hhmm
# curr is only letters
# tframe is e.g. 5m , 2h, 1D

def return_train_from_liste(types_list, from_date, to_date, from_h, to_h, curr, tframe):

    # go in data/no_labels
    # and retrieve all txt files named from the types_list AND that complies with the 
    # other arguments (given as constraints)
    
    types_list =  ["all_heikin_ashi"]

    files_list = []
    import os

    print(os.getcwd())
    #

    for filename in os.listdir(os.getcwd()+"/data/no_labels"):
        
        ##################################
        # different checks on files names
        ##################################
        valid_file = False
        # check if txt file
        if filename.endswith('.txt'):
            #print(filename)
            #print(filename[:-4].split("__"))

            # check if file is in good format
            list_features = filename[:-4].split("__")
            if len(list_features) == 5:
                # check dates format
                dates_arr = list_features[1].split("_")
                if not  len(dates_arr) == 2:
                # check hours format

                # check tframe format

                # check currency format
            else:
                print("not good number of features in filename")
                break
        else:
            print("not a txt file")
            break


        # if filename.split("__")[0] in types_list:
        #     print(f)
    
        # f = os.path.join(os.getcwd()+"/data/no_labels", filename)
    return


return_train_from_liste("types_list", "from_date", "to_date", "from_h", "to_h", "curr", "tframe")