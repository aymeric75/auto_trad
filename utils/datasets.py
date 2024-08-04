


import numpy as np

from itertools import combinations



valid_tf_list = ["1m", "5m", "15m", "1h", "4h"]


def check_filename_valid(filename):


    # check if txt file
    if filename.endswith('.txt'):

        # check if file is in good format (must have 5 "features")
        list_features = filename[:-4].split("__")
        if len(list_features) == 5:

            # check dates format
            dates_arr = list_features[1].split("_")
            if len(dates_arr) == 2:
                if not re.fullmatch(r'^\d{6}$', dates_arr[0]) and re.fullmatch(r'^\d{6}$', dates_arr[1]):
                    return False
            else:
                return False
            # check hours format
            hours_arr = list_features[2].split("_")
            if len(hours_arr) == 2:
                if not re.fullmatch(r'^\d{4}$', hours_arr[0]) and re.fullmatch(r'^\d{4}$', hours_arr[1]):
                    return False

            else:
                return False


            # check currency format
            curr_ = list_features[3]

            # check tframe format
            tframe = list_features[4]

            if tframe not in valid_tf_list:
                return False
            
        else:
            return False
    else:
        return False

    return True


# filename must include ".txt" extension !!
def check_filename_complies(filename, from_date, to_date, from_h, to_h, curr, tframe):

    list_features = filename[:-4].split("__")

    dates_arr = list_features[1].split("_")

    hours_arr = list_features[2].split("_")

    curr_ = list_features[3]

    tframe_ = list_features[4]


    if from_date == dates_arr[0] and to_date == dates_arr[1] and \
        from_h == hours_arr[0] and to_h == hours_arr[1] and \
        curr == curr_ and tframe == tframe_:

        return True
    
    return False


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
       
        if check_filename_valid(filename):
                
            if check_filename_complies(filename):
                print("filename is valid")
                data = np.loadtxt(os.getcwd()+"/data/no_labels/"+filename)
                #print(data.shape)
                retour_list.append(data)
                retour_list_names.append(filename[:-4])
            else:
                print("filename is NOT valid")



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




def return_all_files_combinations_from_list(files_list, N):
    combinations_of_files = list(combinations(files_list, N))
    return combinations_of_files


# ha__100224_100324__0800_1200__runeusdtp__5m.txt
print(check_filename_complies("ha__100224_100324__0800_1200__runeusdtp__5m.txt", "100224", "100324", "0800", "1200", "runeusdtp", "5m"))


