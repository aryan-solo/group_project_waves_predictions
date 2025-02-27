from colorama import Fore, Back, Style
from colorama import init
from termcolor import colored
import os
import sys
from models import *
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
### call from main and then route request for train and test

typ_for_models = ['waveheight','waveperiod']

all_models = {"model_1":{"type":"sequential",
                         "horizon":1,
                         "window_size":7,
                         "activation":"relu",
                         "epoch":100,
                         "function_to_call":"model1_trainer(train_windows, test_windows, train_labels,test_labels,model_name,hyper_paramas)"}
             }

WINDOW_SIZE = 7
HORIZON = 1
def get_train_test_splits(timesteps,timedata,extra):

    full_windows, full_labels = make_windows(timedata, window_size=WINDOW_SIZE, horizon=HORIZON)
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)

    return train_windows, test_windows, train_labels, test_labels


def do_train_test_if_model_does_not_exit(df,station_name):

    for typ in typ_for_models:

        for model in all_models.keys():
            choice = -1
            model_name = f"{model}_{station_name}_{typ}"
            timesteps = df.index.to_numpy()
            timedata = df[typ].to_numpy()
            extra = {}
            train_windows, test_windows, train_labels, test_labels  = get_train_test_splits(timesteps,timedata,extra)
            if os.path.exists(os.path.join(os.getcwd(), 'model_experiments',model_name)):
                prompt = "A trained model already exit , would you like to train again or use existing model,\n1 : train again \n0 : test existing\n"
                choice = input(prompt)

            if int(choice) == 1:
                print(colored(f"starting to train and test for {station_name}, model : {model} for {typ}", 'red', 'on_white'))
                model_params = all_models.get(model)
                hyper_paramas = model_params.get("")
                fn_to_call = model_params["function_to_call"]

                res = eval(fn_to_call)
                sys.exit()
            else:
                print(colored(f"you will be prompted with existing model paramteres for : {station_name}, model : {model} for {typ}", 'blue', 'on_white'))
                sys.exit()

