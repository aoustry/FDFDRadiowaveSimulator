# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 17:17:47 2018

@author: oustry
"""

from FDFDRadiowaveSimulator import FDFDRadiowaveSimulator
from pandas import DataFrame,read_csv

def FirstExample():
    """
    First example of use of the FDFDRadiowaveSimulator class. Generate a .png file

    Returns
    -------
    None.

    """
    mapname = "MAP4"
    sim = FDFDRadiowaveSimulator(mapname)
    #Reading map parameters
    param = read_csv("sources/"+mapname+".csv")
    sim.set_parameters(float(param["dx"]),0.5*float(param["lambda"]),float(param["opt_ind_walls"]),float(param["alpha_walls"]))
    #Generates a bitmap in the output folder describing the field
    Psi = sim.solve(350,125,True)
    


def Generate_Gain_Matrix(mapname,nodes_index):
    """
    Generates the gain matrix associated to a map and a set of sources

    Parameters
    ----------
    mapname : string
        The map name.
    nodes_index : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    #Simulator
    sim = FDFDRadiowaveSimulator(mapname)
    
    #Reading map parameters in the corresponding files (in the "sources" folder)
    param = read_csv("sources/"+mapname+".csv")
    sim.set_parameters(float(param["dx"]),float(param["lambda"]),float(param["opt_ind_walls"]),float(param["alpha_walls"]))
    
    #Reading clients and candidates positions in the corresponding files (in the "sources" folder)
    dataframe_candidates,dataframe_clients = read_csv("sources/"+mapname+"_CA_"+str(nodes_index)+".csv"),read_csv("sources/"+mapname+"_CL_"+str(nodes_index)+".csv")
    list_candidates = [(dataframe_candidates['X'][i],dataframe_candidates['Y'][i]) for i in range(len(dataframe_candidates))]
    list_clients = [(dataframe_clients['X'][i],dataframe_clients['Y'][i]) for i in range(len(dataframe_clients))]
    floor = [dataframe_clients['Floor'][i] for i in range(len(dataframe_clients))] + [dataframe_candidates['Floor'][i] for i in range(len(dataframe_candidates))]
    
    #Computing 2D gain matrix
    gain = sim.gain_matrix(list_clients + list_candidates)
    
    #Applying floor attenuation factor (2.5D method)
    rho = float(param["rho"])
    for i in range(len(gain)):
     	for j in range(len(gain)):
              d=abs(floor[i]-floor[j])
              gain[i,j] = gain[i,j]*(rho**d)
            
    #Storing the gain matrix
    G = DataFrame(gain)
    G.to_csv("output/"+mapname+"_GainMatrix_"+str(nodes_index)+".csv")


if __name__ == "__main__":
   
    FirstExample()
    # MAPLIST = ["MAP1","MAP2","MAP3","MAP4","MAP5","MAP6"]
    # for mapname in MAPLIST:
    #      for i in range(3):
    #          print(mapname,i)
    #          Generate_Gain_Matrix(mapname,i)
        
