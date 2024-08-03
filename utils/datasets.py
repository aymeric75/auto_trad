

#  Une fois que tu as une liste de array où chaque array represent un type 
#
#
#   TU VAS 


import numpy as np




# *_date is ddmmyy
# *_h is hhmm
# curr is only letters
# tframe is e.g. 5m , 2h, 1D
#
#  returns a list of arrays
#  where each array represent a type of data in the
#  given curr, dates etc.

def return_train_from_liste(types_list, from_date, to_date, from_h, to_h, curr, tframe):

    # go in data/no_labels
    # and retrieve all txt files named from the types_list AND that complies with the 
    # other arguments (given as constraints)
    
    types_list =  ["ha"]

    # "ha__100224_100324__0800_1200__runeusdtp__5m"

    retour_list = []

    retour_list_names = []


    files_list = []

    import os
    import re

    print(os.getcwd())
 
    # Loop over the txt files
    for filename in os.listdir(os.getcwd()+"/data/no_labels"):
        
        #print("current file is {}".format(filename))

        ##################################
        # different checks on files names
        ##################################
        valid_file = True

        # check if txt file
        if filename.endswith('.txt'):

            # check if file is in good format (must have 5 "features")
            list_features = filename[:-4].split("__")
            if len(list_features) == 5:

                # check dates format
                dates_arr = list_features[1].split("_")
                if len(dates_arr) == 2:
                    if not re.fullmatch(r'^\d{6}$', dates_arr[0]) and re.fullmatch(r'^\d{6}$', dates_arr[1]):
                        valid_file  = False
                        #print("dates not in the good format")
                        continue
                else:
                    valid_file  = False
                    #print("not good number of dates")
                    continue

                # check hours format
                hours_arr = list_features[2].split("_")
                if len(hours_arr) == 2:
                    if not re.fullmatch(r'^\d{4}$', hours_arr[0]) and re.fullmatch(r'^\d{4}$', hours_arr[1]):
                        valid_file  = False
                        #print("hours not in the good format")
                        continue
                else:
                    valid_file  = False
                    #print("not good number of hours")
                    continue

                # check currency format
                curr_ = list_features[3]

                # check tframe format
                tframe = list_features[4]

                if tframe not in ["1m", "5m", "15m", "1h", "4h"]:
                    valid_file  = False
                    #print("timeframe not in the good format")
                    continue
                
            else:
                valid_file = False
                print("not good number of features in filename")
                continue
        else:
            valid_file = False
            print("not a txt file")
            continue

        if valid_file:
            print("filename is valid")
            data = np.loadtxt(os.getcwd()+"/data/no_labels/"+filename)
            #print(data.shape)
            retour_list.append(data)
            retour_list_names.append(filename[:-4])
        else:
            print("filename is NOT valid")

    #print(retour_list)

    return retour_list, retour_list_names


# entry: two numpy arrays of shape (n_samples, nb_features)
def from_list_of_datas_and_names_save_train_data(list_of_datas, corresponding_names):

    print(corresponding_names)

    n_samples, n_timesteps = list_of_datas[0].shape

    data = np.stack(tuple(list_of_datas), axis=-1) 

    thename=""
    therest = ""
    for i, name in enumerate(corresponding_names):
        if i == 0:
            therest = "__".join(tuple(name.split("__")[1:]))
        thename += name.split("__")[0] + "-"

    thename = thename[:-1] + "__" + therest
    # print("qssssssss")
    # print(data.shape) # (975, 25, 3)

    np.savez(thename+'.npz', data=data)
    return

datas, names = return_train_from_liste("types_list", "from_date", "to_date", "from_h", "to_h", "curr", "tframe")

print(len(datas))
print(names)

from_list_of_datas_and_names_save_train_data(datas, names)