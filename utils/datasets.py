import numpy as np
import os
from itertools import combinations
import re

### DIFFERENCIE BIEN WITH et NO

valid_tf_list = ["1m", "5m", "15m", "1h", "4h"]


def check_filename_valid(filename):


    if not (filename.split("__")[0] == "no" or filename.split("__")[0] == "with"):
        print("file name must start with 'with' or 'no' ")
        return

    if filename.split("__")[0] == "no":
        filename = filename[4:]
    else:
        filename = filename[6:]


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

    if not (filename.split("__")[0] == "no" or filename.split("__")[0] == "with"):
        print("file name must start with 'with' or 'no' ")
        return

    if filename.split("__")[0] == "no":
        filename = filename[4:]
    else:
        filename = filename[6:]

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


#  from thedire 
#  returns a list of arrays
#  where each array represents a type of data in the
#  given curr, dates etc.

def return_dico_from_criteria(thedire, from_date, to_date, from_h, to_h, curr, tframe):

    # go in data/no_labels
    # and retrieve all txt files named from the types_list AND that complies with the 
    # other arguments (given as constraints)
    
    # "ha__100224_100324__0800_1200__runeusdtp__5m"


    retour_list_names = []


    files_list = []

    import os
    import re

 
    dico = {}

    # Loop over the txt files
    for filename in os.listdir(thedire):
        
        #print("current file is {}".format(filename))

        ##################################
        # different checks on files names
        ##################################
       
        if check_filename_valid(filename):
                
            if check_filename_complies(filename, from_date, to_date, from_h, to_h, curr, tframe):
                print("filename is valid")
                data = np.loadtxt(thedire+"/"+filename)
                #print(data.shape)

                dico[filename[:-4]] = data
            else:
                print("filename is NOT valid")



    return dico



# entry: two numpy arrays of shape (n_samples, nb_features)
def from_list_of_datas_and_names_return_in_good_format(list_of_datas):


    n_samples, n_timesteps = list_of_datas[0].shape

    data = np.stack(tuple(list_of_datas), axis=-1) 

    #np.savez(thename+'.npz', data=data)
    return data



def return_all_files_combinations_from_list(files_list, N):
    combinations_of_files = list(combinations(files_list, N))
    return combinations_of_files


def from_criteria_return_list_of_train_test(dico_trains, dico_tests):



    # 
    files_combi_trains = return_all_files_combinations_from_list(list(dico_trains.keys()), 2)
    
    files_combi_tests = return_all_files_combinations_from_list(list(dico_tests.keys()), 2)

    return files_combi_trains, files_combi_tests



def all_train_and_test_data():

    # useful data
    dico_trains = return_dico_from_criteria("/workspace/auto_trad"+"/data/no_labels","100224", "100324", "0800", "1200", "runeusdtp", "5m")

    dico_tests = return_dico_from_criteria("/workspace/auto_trad"+"/data/with_labels","100224", "100324", "0800", "1200", "runeusdtp", "5m")

    # 1) return 2 lists: one of the train files, one of the <=> test files
    files_combi_trains, files_combi_tests = from_criteria_return_list_of_train_test(dico_trains, dico_tests)


    dico_of_train_test_combis = {}

    ## 3) for each combination for "style" in files_combi_trains
    for combi in files_combi_trains:

        # making the name for the combi
        name_combi = ""
        styles = []
        for iii in range(len(combi)):
            styles.append(combi[iii].split("__")[1])
        name_combi += "__".join(combi[0].split("__")[2:])
        name_combi = "__".join(styles) + "__" + name_combi

        all_corres_tests = True
        for iii in range(len(combi)):
            if "with__"+combi[iii][4:] not in list(dico_tests.keys()):
                all_corres_tests = False

        train_files = []
        test_files = []

        # all_corres_tests
        for iii in range(len(combi)):
            # TYPE iii of the combi
            train_files.append(dico_trains[combi[iii]])
            test_files.append(dico_tests["with__"+combi[iii][4:]])

        data_train = from_list_of_datas_and_names_return_in_good_format(train_files)
        data_test = from_list_of_datas_and_names_return_in_good_format(test_files)

        dico_of_train_test_combis[name_combi] = {
            "train" : data_train,
            "test" : data_test
        }
        

    return dico_of_train_test_combis



# return_dico_from_criteria(os.getcwd()+"/data/no_labels", types_list, from_date, to_date, from_h, to_h, curr, tframe)
