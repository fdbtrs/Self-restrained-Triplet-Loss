import matplotlib
matplotlib.use('agg')
from pyeer import plot
from pyeer.eer_info import get_eer_stats

import os
import numpy as np


##
##This script generates a plot for maskfilm Baseline
##

#local setting, depends on name of files and modeltype
cases = ["BLR-BLP", "BLR-M12P", "BLR-M12PTriplet", "BLR-M12PSRT", "M12R-M12P",    "M12R-M12PTriplet" , "M12RSRT-M12PSRT"]
#label of the curves
plot_names = ["BLR-BLP", "BLR-MP",  "BLR-MP(T)", "BLR-MP(SRT)", "MR-MP","MR-MP(T)","MR-MP(SRT)"]



cases = ["BLR-M12P", "BLR-M12PTriplet", "BLR-M12PSRT"]
#label of the curves
plot_names = ["BLR-MP", "BLR-MP(T)", "BLR-MP(SRT)"]
'''

cases = [ "M12R-M12P",    "M12R-M12PTriplet" , "M12RSRT2-M12PSRT2"]
#label of the curves
plot_names = [ "MR-MP","MR-MP(T)","MR-MP(SRT)"]
'''

#target folder
roc_path = './plots/roc_log_cos_um'
table_path = "./plots/tables_cos_um"
#models'similarities_cosine/ResNet50',
models = [ 'similarities_cosine/ResNet100']
models = [ 'similarities_cosine/ResNet100']

#set true if you want to create tables
table_creation = True

#directory of this file, used to jump back in directory for next model
file_path = os.path.dirname(os.path.realpath(__file__))
    
#convert genuine|imposter content to list of floats
def getdata(path):
    res = []
    with open(path, 'r') as file:
        for i, line in enumerate(file):
            res.append(float(line))
    return res  
    
#defines first lines of a new table
def new_table(model, name):
    table = open("./"+ name + ".txt", "w")
    table.write("\\begin{table}[]\n")
    table.write("\\begin{tabular}{|l|l|l|l|l|l|l|}\n")
    table.write("\hline\n")
    table.write(model+" & EER & FMR100 & FMR1000 & G-mean & I-mean & FDR \\\ \hline\n")

    return table

# iterate over models
for model in models:
    model_type = model.split('/')[1]
    if model_type == 'Mobilefacenet':
        model_name = "Mobilefacenet"
    elif model_type == 'ResNet100':
        model_name = "ResNet100"
    else:
        model_name = "Resnet50"
        
    target_path = roc_path + '_' + model_type
    #create target folder, where all plots will be stored
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    #create target subfolder
    target_path = os.path.join(target_path, 'Baseline')
    if not os.path.exists(target_path):
        os.mkdir(target_path)
        
    #table target location
    table_target_path = table_path + '_' + model_type
    #create target folder, where all tables will be stored
    if not os.path.exists(table_target_path):
        os.mkdir(table_target_path)
      
    fdrs = []
    all_stats = []
    for case in cases:
        #generate stats for plotting
        genuine = getdata("./%s/%s/genuine.txt" % (model, case))
        imposter = getdata("./%s/%s/imposter.txt" % (model, case))
        stats = get_eer_stats(genuine, imposter)
        #compute FDR
        fdr = ((stats.gmean-stats.imean)**2) / (stats.gstd**2 + stats.istd**2)
        fdrs.append(fdr)
        #append stats
        all_stats.append(stats)
    #switch to target folder
    os.chdir("./%s" % (target_path))
    print(cases)
    print("--------------------------------------------------------------------")
    print(plot_names)
    #plot all stats with labels
    plot_name = "_maskfilm_%s_" % model_type + "Baseline"
    plot.plt_roc_curve(all_stats, plot_names, save_path="./", dpi=1200,ext=plot_name+'.eps')
        
    plot_name = "_mfr2_%s_" % model_type + "Baseline_Distribution"

    plot.plt_distributions(all_stats, plot_names, save_path="./", dpi=1200,ext=plot_name+'.eps')

    if table_creation:
        #switch to table target folder
        os.chdir(os.path.join(file_path, table_target_path))
        #create table
        table = new_table(model_name, "Baseline")
        # filter best results
        best_eer = 1
        best_fmr100 = 1
        best_fmr1000 = 1
        best_FDR = 0
        for index, state in enumerate(all_stats):
            if best_eer > state.eer:
                best_eer = state.eer
            if best_fmr100 > state.fmr100:
                best_fmr100 = state.fmr100
            if best_fmr1000 > state.fmr1000:
                best_fmr1000 = state.fmr1000
            if best_FDR < fdrs[index]:
                best_FDR = fdrs[index]
                
        for index, st in enumerate(all_stats):
            # compute percentage values
            # compute percentage
            eer = st.eer
            eer *= 100
            eer = round(eer, 4)
            fmr100 = st.fmr100
            fmr100 *= 100
            fmr100 = round(fmr100, 4)
            fmr1000 = st.fmr1000
            fmr1000 *= 100
            fmr1000 = round(fmr1000, 4)
            FDR = round(fdrs[index], 4)
            # set best values
            if st.eer == best_eer:
                eer = "\\textbf{" + str(eer) + "}"
            else:
                eer = str(eer)
            if st.fmr100 == best_fmr100:
                fmr100 = "\\textbf{" + str(fmr100) + "}"
            else:
                fmr100 = str(fmr100)
            if st.fmr1000 == best_fmr1000:
                fmr1000 = "\\textbf{" + str(fmr1000) + "}"
            else:
                fmr1000 = str(fmr1000)
            if fdrs[index] == best_FDR:
                FDR = "\\textbf{" + str(FDR) + "}"
            else:
                FDR = str(FDR)
                
            # experiment, EER, FMR100, FMR1000, G-mean, I-mean, FDR
            table.write(plot_names[index] + " & " + eer + "\% & " + fmr100 + "\% & " + fmr1000 + "\% & " + str(round(st.gmean, 4)) + " & " + str(round(st.imean, 4)) + " & " + FDR + " \\\ \hline\n")
            print(plot_names[index] + " & " + eer + "\% & " + fmr100 + "\% & " + fmr1000 + "\% & " + str(round(st.gmean, 4)) + " & " + str(round(st.imean, 4)) + " & " + FDR + " \\\ \hline\n")

        # add last lines of latex table
        table.write("\end{tabular}\n")
        table.write("\caption{Maskfilm-Baseline}\n")
        table.write("\end{table}")
        table.close()
    #jump back to root folder
    os.chdir(file_path)
