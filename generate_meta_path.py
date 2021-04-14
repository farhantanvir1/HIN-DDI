#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:18:07 2020

@author: illusionist
"""
from csv import DictReader
import numpy as np
import pandas as pd
import xlrd
import csv
import os
import csv
import xlrd
import _thread
import time
import torch as th
from sklearn.preprocessing import StandardScaler
import rdkit
from rdkit import Chem
from rdkit.Chem import MACCSkeys

"""
    this function creates a dictionary for entity(drug/protein/species/indication) sent as argument
"""

def create_dictionary(path):
    
    count=1
    dict_val={}
    # dictionary's key will content entity object's id(drug id/protein id)
    # value will contain the index id where corresponding entity object's data will be found in adjacency and
    # commuting matrix
    with open(path) as f:
        content = f.readlines()
        for x in content:
            dict_val[x.strip()]=count
            count+=1
    
    
    return dict_val

"""
    this function creates adjacency matrix for 2 entities' dictionaries sent as arguments
    it has 8 parameters. 
    first parameter refers to file path
    
    second and third parameter refers to first entity's desired column name in file(where 
    we will look for values of first entity) and first entity's dictonary
    
    forth and fifth parameter refers to second entity's desired column name in file(where 
    we will look for values of second entity) and second entity's dictonary
    
    sixth abd seventh parameter indicates whether the entities are drugs.
    drug data maybe split by semicolons.
    if it is, we have to do some text processing on drug ids. only then, we can obtain accurate index id 
    from drug's dictionary.
    
    
"""


def create_adjacency_matrix_labelled_data(path, col1, col2, dict1, FirstColHasMultiValue, SecondColHasMultiValue):
    
    adj_mat=[[]]
    
    #initializing all values in adjacency matrix to 0
    #getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict1.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        csv_dict_reader = DictReader(read_obj)
        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:
            # row variable is a dictionary that represents a row in csv
            
            #considering different combinations when columns maybe semicolon-seperated or not
            #for each case, null checking is done
            if FirstColHasMultiValue==False and SecondColHasMultiValue==False:
                if row[col1]!='':
                    adj_mat[dict1[row[col1]]][dict1[row[col2]]]=1
                    adj_mat[dict1[row[col2]]][dict1[row[col1]]]=1
            elif FirstColHasMultiValue==True and SecondColHasMultiValue==False:
                if row[col1]!='':
                    items=row[col1].split('; ')
                    for item in items:
                        adj_mat[dict1[item]][dict1[row[col2]]]=1
                        adj_mat[dict1[row[col2]]][dict1[item]]=1
            elif FirstColHasMultiValue==False and SecondColHasMultiValue==True:
                if row[col1]!='' and row[col2]!='':
                    items=row[col2].split('; ')
                    for item in items:
                        if item in dict1.keys():
                            adj_mat[dict1[row[col1]]][dict1[item]]=1
                            adj_mat[dict1[item]][dict1[row[col1]]]=1
             
            elif FirstColHasMultiValue==True and SecondColHasMultiValue==True:
                if row[col1]!='' and row[col2]!='':
                    items_1=row[col1].split('; ')
                    items_2=row[col2].split('; ')
                    for item_1 in items_1:
                         for item_2 in items_2:
                             adj_mat[dict1[item_1]][dict1[item_2]]=1
                             adj_mat[dict1[item_2]][dict1[item_1]]=1

    return adj_mat

"""
    this function creates adjacency matrix for 2 entities' dictionaries sent as arguments
    it has 7 parameters. 
    first parameter refers to file path
    
    second and third parameter refers to first entity's desired column name in file(where 
    we will look for values of first entity) and first entity's dictonary
    
    forth and fifth parameter refers to second entity's desired column name in file(where 
    we will look for values of second entity) and second entity's dictonary
    
    sixth and seventh parameter indicates whether first and second colummn values have multiple values.
    in case of multiple values, values maybe split by semicolons.
    if it is, we have to do some text processing.
    
    sixth parameter indicates whether the first entity is drug.
    drug data maybe split by semicolons.
    if it is, we have to do some text processing on drug ids. only then, we can obtain accurate index id 
    from drug's dictionary.
    
    seventh parameter indicates whether the meta-path, adjacency matrix is desired for, 
    is of ADE/Indication relationship or not
    
    if the relationship type is ADE/Indication, then the eighth parameter refers to the type of relationship
    
"""

def create_adjacency_matrix(path, col1, dict1, col2, dict2, FirstColHasMultiValue, SecondColHasMultiValue):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
 
        
        csv_dict_reader = DictReader(read_obj)

        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:

            # considering different combinations when columns maybe semicolon-seperated or not
            # for each case, null checking is done
            if FirstColHasMultiValue==False:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        adj_mat[dict1[row[col1]]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        items=row[col2].split('; ')
                        for item in items:
                            adj_mat[dict1[row[col1]]][dict2[item]]=1
            elif FirstColHasMultiValue==True:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        items=row[col1].split('; ')  
                        adj_mat[dict1[item]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        items1=row[col1].split('; ')
                        items2=row[col1].split('; ')
                        for item1 in items1:
                            for item2 in items2:
                                adj_mat[dict1[item1]][dict2[item1]]=1
            """
            elif isDrug1==False and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items=row[col2].split('; ')
                    for item in items:
                        adj_mat[dict1[row[col1]]][dict2[item]]=1
             
            elif isDrug1==True and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items_1=row[col1].split('; ')
                    items_2=row[col2].split('; ')
                    for item_1 in items_1:
                         for item_2 in items_2:
                             adj_mat[dict1[item_1]][dict2[item_2]]=1
            """

    return adj_mat



"""
    this function creates adjacency matrix for drug-pathway
    it has 5 parameters. 
    first parameter refers to file path
    
    second and third parameter refers to first entity's desired column name in file(where 
    we will look for values of first entity) and first entity's dictonary
    
    forth and fifth parameter refers to second entity's desired column name in file(where 
    we will look for values of second entity) and second entity's dictonary
    
    sixth and seventh parameter indicates whether first and second colummn values have multiple values.
    in case of multiple values, values maybe split by semicolons.
    if it is, we have to do some text processing.
    
    
"""


def create_adjacency_matrix_2(path, dict1, ind1, dict2, ind2):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[ind2].strip() != '' and row[ind1].strip() != '':
                row[ind1] = row[ind1].replace('\ufeff','').strip()
                row[ind2] = row[ind2].replace('\ufeff','').strip()
                adj_mat[dict1[row[ind1]]][dict2[row[ind2].strip()]]=1
                
    return adj_mat



"""
    this function creates adjacency matrix for 2 entities' dictionaries sent as arguments
    it has 7 parameters. 
    first parameter refers to file path
    
    second and third parameter refers to first entity's desired column name in file(where 
    we will look for values of first entity) and first entity's dictonary
    
    forth and fifth parameter refers to second entity's desired column name in file(where 
    we will look for values of second entity) and second entity's dictonary
    
    sixth and seventh parameter indicates whether first and second colummn values have multiple values.
    in case of multiple values, values maybe split by semicolons.
    if it is, we have to do some text processing.
    
    eighth parameter refers to the number of characters till which atc code of drug have to be processed
    
"""

def create_adjacency_matrix_atc_code(path, col1, dict1, col2, dict2, number):
    
    flag=False
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
 
        
        csv_dict_reader = DictReader(read_obj)

        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:

            if row[col1]!='' and row[col2]!='':
                items=row[col2].split('; ')
                for item in items:
                    if item[0:2]=="QA":
                        number+=1
                        flag=True
                    adj_mat[dict1[row[col1]]][dict2[item[0:number]]]=1
                    if flag==True:
                        number-=1
                        flag=False

    return adj_mat


def create_adjacency_matrix_drug_ade_ind(path, col1, dict1, col2, dict2, FirstColHasMultiValue, SecondColHasMultiValue):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, max(dict2.values())+1)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
 
        
        csv_dict_reader = DictReader(read_obj)

        # iterate over each line as a ordered dictionary
        for row in csv_dict_reader:

            # considering different combinations when columns maybe semicolon-seperated or not
            # for each case, null checking is done
            if FirstColHasMultiValue==False:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        adj_mat[dict1[row[col1]]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        if row[col2].find('\n') != -1:
                            items=row[col2].split(';\n')
                            for item in items:
                                adj_mat[dict1[row[col1]]][dict2[item]]=1
                        elif row[col2].find('\n') == -1:
                            items=row[col2].split(';')
                            for item in items:
                                adj_mat[dict1[row[col1]]][dict2[item]]=1
            elif FirstColHasMultiValue==True:
                if SecondColHasMultiValue==False:
                    if row[col1]!='' and row[col2]!='':
                        items=row[col1].split(';')  
                        adj_mat[dict1[item]][dict2[row[col2]]]=1
                elif SecondColHasMultiValue==True:
                    if row[col1]!='' and row[col2]!='':
                        items1=row[col1].split(';')
                        items2=row[col1].split(';')
                        for item1 in items1:
                            for item2 in items2:
                                adj_mat[dict1[item1]][dict2[item1]]=1
            """
            elif isDrug1==False and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items=row[col2].split('; ')
                    for item in items:
                        adj_mat[dict1[row[col1]]][dict2[item]]=1
             
            elif isDrug1==True and isDrug2==True:
                if row[col1]!='' and row[col2]!='':
                    items_1=row[col1].split('; ')
                    items_2=row[col2].split('; ')
                    for item_1 in items_1:
                         for item_2 in items_2:
                             adj_mat[dict1[item_1]][dict2[item_2]]=1
            """

    return adj_mat

"""
    this function is responsible for generation of meta-path
    we construct meta-paths through matrix multiplication

"""

def create_adjacency_matrix_chemical_structure(path, col1, col2, dict1):
    
    adj_mat=[[]]
    
    # initializing all values in adjacency matrix to 0
    # getting max index values from dictionaries to set row and col values in adjacency matrix
    rows, cols= (max(dict1.values())+1, 167)
    adj_mat = [[0] * cols for i in range(rows)]
    
    #dfs = pd.read_excel(path, sheetname=None)
    
    # if dictionaries used as arguments have an association, their corresponding cell's value in 
    # adjacency matrix will be set to 1. otherwise, the value will be set to 0.
    with open(path, 'r', encoding='latin-1') as read_obj:
    # pass the file object to DictReader() to get the DictReader object
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            for i in range(167):
                adj_mat[dict1[row[col1]]][i]=int(row[col2][i+1])

    return adj_mat

def generate_meta_path(mat, length, dimension):
    
    commuting_matrix = [[0] * dimension for i in range(dimension)]
    count=1
    resultant_matrix=mat[0]
    
    for i in range(length-1):
        x = th.FloatTensor(resultant_matrix)
        y = th.FloatTensor(mat[i+1])
        resultant_matrix = th.mm(x,y)
        #resultant_matrix = np.matmul(resultant_matrix, mat[i+1])
    
    
    return resultant_matrix.numpy()

"""
    this function is required for calculation of topological features

"""
    
def create_connectivity_matrix(mat, dimension):
    
    starts_with_connectivity_mat=[0] * dimension
    ends_with_connectivity_mat=[0] * dimension

    starts_with_connectivity_mat = np.sum(mat, axis = 0)
    ends_with_connectivity_mat = np.sum(mat, axis = 1)

    return starts_with_connectivity_mat, ends_with_connectivity_mat

"""

"""

"""
    this function computes topological features, i.e., path count, normalized path count, 
    random walk, symmetric random walk

"""
def compute_topological_features(PC, starts_with_PC, end_with_PC, dimension):
    
    NPC = [[0.00] * dimension for i in range(dimension)]
    RW = [[0.00] * dimension for i in range(dimension)]
    SRW = [[0.00] * dimension for i in range(dimension)]
    den1=0
    den2=0
    
    for i in range(dimension):
        for j in range(dimension):
            if i!= j :
                if starts_with_PC[i]!=0 or end_with_PC[j]!=0:
                    if starts_with_PC[i]==0 and end_with_PC[j]!=0:
                        NPC[i][j] = (PC[i][j] + PC[j][i])/(end_with_PC[j])
                    elif starts_with_PC[i]!=0 and end_with_PC[j]==0:
                        NPC[i][j] = (PC[i][j] + PC[j][i])/(starts_with_PC[i])
                    else:
                        NPC[i][j] = (PC[i][j] + PC[j][i])/(starts_with_PC[i]+end_with_PC[j])
                if starts_with_PC[i]!=0:
                    RW[i][j] = PC[i][j] / starts_with_PC[i]
                if starts_with_PC[i]!=0 or starts_with_PC[j]!=0:
                    if starts_with_PC[i]==0 or starts_with_PC[j]!=0:
                        SRW[i][j] = PC[j][i] / starts_with_PC[j]
                    elif starts_with_PC[i]!=0 or starts_with_PC[j]==0:
                        SRW[i][j] = PC[i][j] / starts_with_PC[i]
                    else:
                        SRW[i][j] = (PC[i][j] / starts_with_PC[i]) + (PC[j][i] / starts_with_PC[j])
    return NPC, RW, SRW
    

# create dictionaries for each entities        
drugs_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ids.txt')
species_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/species 2.txt')
protein_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/protein_ids.txt')

pathway_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/pathway_ids.txt')
pathway_subjects_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/pathway_subjects.txt')
CUI_dict=CUI_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/CUI_ids.txt')

atc_1st_level_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/ATC_code_1st level.txt')
atc_2nd_level_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/ATC_code_2nd_level.txt')
atc_3rd_level_dict=create_dictionary('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/ATC_code_3rd_level.txt')

# adjacency matrices for drug- protein
drug_protein_adj_mat = create_adjacency_matrix('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'target proteins', protein_dict, False, True)

# adjacency matrices for drug- pathway
drug_pathway_adj_mat = create_adjacency_matrix('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'pathways', pathway_dict, False, True)

# adjacency matrices for drug- ATC code
drug_atc_1st_adj_mat = create_adjacency_matrix_atc_code('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'atc code', atc_1st_level_dict, 1)
drug_atc_2nd_adj_mat = create_adjacency_matrix_atc_code('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'atc code', atc_2nd_level_dict, 3)
drug_atc_3rd_adj_mat = create_adjacency_matrix_atc_code('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'atc code', atc_3rd_level_dict, 4)

# adjacency matrices for drug- indication and ADE
drug_ade_adj_mat = create_adjacency_matrix_drug_ade_ind('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'Side effect', CUI_dict, False, True)
drug_ind_adj_mat = create_adjacency_matrix_drug_ade_ind('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', drugs_dict, 'indication', CUI_dict, False, True)

drug_che_struc_adj_mat = create_adjacency_matrix_chemical_structure('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', 'fingerprint', drugs_dict)

#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_pathway_adj_mat.txt", drug_pathway_adj_mat)

#drug_ade_adj_mat, drug_ind_adj_mat = create_adjacency_matrix_drug_ade_ind('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/DEB2.xlsx', drugs_dict, 1, CUI_dict, 2)

# adjacency matrices for protein - species and pathway - subject/category
protein_species_adj_mat = create_adjacency_matrix_2('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/protein_organism.csv', protein_dict, 0, species_dict, 1)

pathway_subject_adj_mat = create_adjacency_matrix_2('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/pathway_category.csv', pathway_dict, 0, pathway_subjects_dict, 1)




# adjacency matrices for species - protein/enzyme/carrier/transporter and protein/enzyme/carrier/transporter - drug
species_protein_adj_mat = [[protein_species_adj_mat[j][i] for j in range(len(protein_species_adj_mat))] for i in range(len(protein_species_adj_mat[0]))] 
protein_drug_adj_mat = [[drug_protein_adj_mat[j][i] for j in range(len(drug_protein_adj_mat))] for i in range(len(drug_protein_adj_mat[0]))]



pathway_drug_adj_mat = [[drug_pathway_adj_mat[j][i] for j in range(len(drug_pathway_adj_mat))] for i in range(len(drug_pathway_adj_mat[0]))] 
subject_pathway_adj_mat = [[pathway_subject_adj_mat[j][i] for j in range(len(pathway_subject_adj_mat))] for i in range(len(pathway_subject_adj_mat[0]))] 



# adjacency matrices for ADE/Indication - drug
ade_drug_adj_mat = [[drug_ade_adj_mat[j][i] for j in range(len(drug_ade_adj_mat))] for i in range(len(drug_ade_adj_mat[0]))] 
ind_drug_adj_mat = [[drug_ind_adj_mat[j][i] for j in range(len(drug_ind_adj_mat))] for i in range(len(drug_ind_adj_mat[0]))]


# adjacency matrices for drug-ATC code
atc_1st_drug_adj_mat = [[drug_atc_1st_adj_mat[j][i] for j in range(len(drug_atc_1st_adj_mat))] for i in range(len(drug_atc_1st_adj_mat[0]))] 
atc_2nd_drug_adj_mat = [[drug_atc_2nd_adj_mat[j][i] for j in range(len(drug_atc_2nd_adj_mat))] for i in range(len(drug_atc_2nd_adj_mat[0]))]
atc_3rd_drug_adj_mat = [[drug_atc_3rd_adj_mat[j][i] for j in range(len(drug_atc_3rd_adj_mat))] for i in range(len(drug_atc_3rd_adj_mat[0]))]

che_struc_drug_adj_mat = [[drug_che_struc_adj_mat[j][i] for j in range(len(drug_che_struc_adj_mat))] for i in range(len(drug_che_struc_adj_mat[0]))] 


print("Labelled data")
Labelled_data = create_adjacency_matrix_labelled_data('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data_drugbank_interacting_drugs_pathway SMILES_indication_atc code-2.csv', 'id', 'interacting drugs', drugs_dict, False, True)



"""
    #generating drug-pathway meta-path
    #this is equivalent to calculating path count of this meta-path
"""

row = max(drugs_dict.values())+1
drug_pathway_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(4)]
drug_pathway_meta_path[0]=drug_pathway_adj_mat
drug_pathway_meta_path[1]=pathway_subject_adj_mat
drug_pathway_meta_path[2]=subject_pathway_adj_mat
drug_pathway_meta_path[3]=pathway_drug_adj_mat


drug_pathway_meta_path = generate_meta_path(drug_pathway_meta_path,4,row)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_pathway_PC.txt", drug_pathway_meta_path)
print('Drug Pathway done')

"""
    #generating drug-protein meta-path
    #this is equivalent to calculating path count of this meta-path
"""
row = max(drugs_dict.values())+1
drug_protein_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(4)]
drug_protein_meta_path[0]=drug_protein_adj_mat
drug_protein_meta_path[1]=protein_species_adj_mat
drug_protein_meta_path[2]=species_protein_adj_mat
drug_protein_meta_path[3]=protein_drug_adj_mat
#drug_protein_meta_path = np.asmatrix(drug_protein_meta_path)
drug_protein_meta_path = generate_meta_path(drug_protein_meta_path,4,row)
print('Drug protein done')
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_protein_PC.txt", drug_protein_meta_path)


"""
    #generating drug-enzyme meta-path
    #this is equivalent to calculating path count of this meta-path
"""
"""
row = max(drugs_dict.values())+1
drug_enzyme_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(4)]
drug_enzyme_meta_path[0]=drug_enzyme_adj_mat
drug_enzyme_meta_path[1]=enzyme_species_adj_mat
drug_enzyme_meta_path[2]=species_enzyme_adj_mat
drug_enzyme_meta_path[3]=enzyme_drug_adj_mat

print('Drug Enzyme')
drug_enzyme_meta_path = generate_meta_path(drug_enzyme_meta_path,4,row)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_enzyme_PC.txt", drug_enzyme_meta_path)


"""
   # generating drug-carrier meta-path
   # this is equivalent to calculating path count of this meta-path
"""


row = max(drugs_dict.values())+1
drug_carrier_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(4)]
drug_carrier_meta_path[0]=drug_carrier_adj_mat
drug_carrier_meta_path[1]=carrier_species_adj_mat
drug_carrier_meta_path[2]=species_carrier_adj_mat
drug_carrier_meta_path[3]=carrier_drug_adj_mat

print('Drug Carrier')
drug_carrier_meta_path = generate_meta_path(drug_carrier_meta_path,4,row)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_carrier_PC.txt", drug_carrier_meta_path)


"""
   # generating drug-transporter meta-path
   # this is equivalent to calculating path count of this meta-path
"""

row = max(drugs_dict.values())+1
drug_transporter_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(4)]
drug_transporter_meta_path[0]=drug_transporter_adj_mat
drug_transporter_meta_path[1]=transporter_species_adj_mat
drug_transporter_meta_path[2]=species_transporter_adj_mat
drug_transporter_meta_path[3]=transporter_drug_adj_mat

print('Drug Transporter')
drug_transporter_meta_path = generate_meta_path(drug_transporter_meta_path,4,row)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_transporter_PC.txt", drug_transporter_meta_path)
"""
"""
  #  generating drug-ADE meta-path
  #  this is equivalent to calculating path count of this meta-path
"""

row = max(drugs_dict.values())+1
drug_ade_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_ade_meta_path[0]=drug_ade_adj_mat
drug_ade_meta_path[1]=ade_drug_adj_mat


drug_ade_meta_path = generate_meta_path(drug_ade_meta_path,2,row)
print('Drug ADE done')
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ade_PC.txt", drug_ade_meta_path)


"""
  #  generating drug-Indication meta-path
  #  this is equivalent to calculating path count of this meta-path
"""


row = max(drugs_dict.values())+1
drug_ind_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_ind_meta_path[0]=drug_ind_adj_mat
drug_ind_meta_path[1]=ind_drug_adj_mat


drug_ind_meta_path = generate_meta_path(drug_ind_meta_path,2,row)
print('Drug Ind done')

"""
  #  generating drug-atc code meta-path
  #  this is equivalent to calculating path count of this meta-path
"""

row = max(drugs_dict.values())+1
drug_atc_1st_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_atc_1st_meta_path[0]=drug_atc_1st_adj_mat
drug_atc_1st_meta_path[1]=atc_1st_drug_adj_mat

drug_atc_1st_meta_path = generate_meta_path(drug_atc_1st_meta_path,2,row)

row = max(drugs_dict.values())+1
drug_atc_2nd_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_atc_2nd_meta_path[0]=drug_atc_2nd_adj_mat
drug_atc_2nd_meta_path[1]=atc_2nd_drug_adj_mat

drug_atc_2nd_meta_path = generate_meta_path(drug_atc_2nd_meta_path,2,row)

row = max(drugs_dict.values())+1
drug_atc_3rd_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_atc_3rd_meta_path[0]=drug_atc_3rd_adj_mat
drug_atc_3rd_meta_path[1]=atc_3rd_drug_adj_mat


drug_atc_3rd_meta_path = generate_meta_path(drug_atc_3rd_meta_path,2,row)
print('Drug atc code done')

"""
  #  generating drug-chemical structure meta-path
  #  this is equivalent to calculating path count of this meta-path
"""
row = max(drugs_dict.values())+1
drug_chem_struct_meta_path = [[[0 for k in range(row)] for j in range(row)] for i in range(2)]
drug_chem_struct_meta_path[0]=drug_che_struc_adj_mat
drug_chem_struct_meta_path[1]=che_struc_drug_adj_mat


drug_chem_struct_meta_path = generate_meta_path(drug_chem_struct_meta_path,2,row)
print('Drug chemical strucuture done')

#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ind_PC.txt", drug_ind_meta_path)



"""
  #  in this block, meta-path topological features are computed

"""

count = max(drugs_dict.values())+1
starts_with_drug_pathway, ends_with_drug_pathway = create_connectivity_matrix(drug_pathway_meta_path, count)
starts_with_drug_protein, ends_with_drug_protein = create_connectivity_matrix(drug_protein_meta_path, count)
#starts_with_drug_enzyme, ends_with_drug_enzyme = create_connectivity_matrix(drug_enzyme_meta_path, count)
#starts_with_drug_carrier, ends_with_drug_carrier = create_connectivity_matrix(drug_carrier_meta_path, count)
#starts_with_drug_transporter, ends_with_drug_transporter = create_connectivity_matrix(drug_transporter_meta_path, count)
starts_with_drug_ade, ends_with_drug_ade = create_connectivity_matrix(drug_ade_meta_path, count)
starts_with_drug_ind, ends_with_drug_ind = create_connectivity_matrix(drug_ind_meta_path, count)
starts_with_drug_atc_1st, ends_with_drug_atc_1st = create_connectivity_matrix(drug_atc_1st_meta_path, count)
starts_with_drug_atc_2nd, ends_with_drug_atc_2nd = create_connectivity_matrix(drug_atc_2nd_meta_path, count)
starts_with_drug_atc_3rd, ends_with_drug_atc_3rd = create_connectivity_matrix(drug_atc_3rd_meta_path, count)
starts_with_drug_chem_struct, ends_with_drug_chem_struct = create_connectivity_matrix(drug_chem_struct_meta_path, count)


#computing topological features based on meta-paths

print("Toplogical features")
print('Drug Pathway')
drug_pathway_NPC, drug_pathway_RW, drug_pathway_SRW = compute_topological_features(drug_pathway_meta_path, starts_with_drug_pathway, ends_with_drug_pathway, count)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_pathway_NPC.txt", drug_pathway_NPC)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_pathway_RW.txt", drug_pathway_RW)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_pathway_SRW.txt", drug_pathway_SRW)    


print('Drug protein')
drug_protein_NPC, drug_protein_RW, drug_protein_SRW = compute_topological_features(drug_protein_meta_path, starts_with_drug_protein, ends_with_drug_protein, count)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_protein_NPC", drug_protein_NPC)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_protein_RW.txt", drug_protein_RW)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_protein_SRW.txt", drug_protein_SRW)    

print('done')
"""
#print('Drug Enzyme')
drug_enzyme_NPC, drug_enzyme_RW, drug_enzyme_SRW = compute_topological_features(drug_enzyme_meta_path, starts_with_drug_enzyme, ends_with_drug_enzyme, count)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_enzyme_NPC.txt", drug_enzyme_NPC)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_enzyme_RW.txt", drug_enzyme_RW)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_enzyme_SRW.txt", drug_enzyme_SRW)  

     
#print('Drug Carrier')
drug_carrier_NPC, drug_carrier_RW, drug_carrier_SRW = compute_topological_features(drug_carrier_meta_path, starts_with_drug_carrier, ends_with_drug_carrier, count)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_carrier_NPC.txt", drug_carrier_NPC)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_carrier_RW.txt", drug_carrier_RW)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_carrier_SRW.txt", drug_carrier_SRW) 

print('Drug Transporter')
drug_transporter_NPC, drug_transporter_RW, drug_transporter_SRW = compute_topological_features(drug_transporter_meta_path, starts_with_drug_transporter, ends_with_drug_transporter, count)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_transporter_NPC.txt", drug_transporter_NPC)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_transporter_RW.txt", drug_transporter_RW)
np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_transporter_SRW.txt", drug_transporter_SRW) 
"""
   
print('Drug ADE')
drug_ade_NPC, drug_ade_RW, drug_ade_SRW = compute_topological_features(drug_ade_meta_path, starts_with_drug_ade, ends_with_drug_ade, count)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ade_NPC.txt", drug_ade_NPC)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ade_RW.txt", drug_ade_RW)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ade_SRW.txt", drug_ade_SRW) 


print('Drug Indication')
drug_ind_NPC, drug_ind_RW, drug_ind_SRW = compute_topological_features(drug_ind_meta_path, starts_with_drug_ind, ends_with_drug_ind, count)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ind_NPC.txt", drug_ind_NPC)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ind_RW.txt", drug_ind_RW)
#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_ind_SRW.txt", drug_ind_SRW)

print('Drug Chemical structure')
drug_chem_NPC, drug_chem_RW, drug_chem_SRW = compute_topological_features(drug_chem_struct_meta_path, starts_with_drug_chem_struct, ends_with_drug_chem_struct, count)

print('Drug ATC Code')
drug_atc_1st_NPC, drug_atc_1st_RW, drug_atc_1st_SRW = compute_topological_features(drug_atc_1st_meta_path, starts_with_drug_atc_1st, ends_with_drug_atc_1st, count)
drug_atc_2nd_NPC, drug_atc_2nd_RW, drug_atc_2nd_SRW = compute_topological_features(drug_atc_2nd_meta_path, starts_with_drug_atc_2nd, ends_with_drug_atc_2nd, count)
drug_atc_3rd_NPC, drug_atc_3rd_RW, drug_atc_3rd_SRW = compute_topological_features(drug_atc_3rd_meta_path, starts_with_drug_atc_3rd, ends_with_drug_atc_3rd, count)




#np.savetxt("/Volumes/Farhan/Research/Code/Meta-path/DDID/data/labelled_data.txt", Labelled_data)





with open('/Volumes/Farhan/Research/Code/Meta-path/DDID/data/drug_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Drug1", "Drug2", "Protein_PC", "Pathway_PC", "ADE_PC", "Indication_PC", "Chemical_Structure_PC", "ATC_1st_PC", "ATC_2nd_PC", "ATC_3rd_PC", "Protein_NPC", "Pathway_NPC", "ADE_NPC", "Indication_NPC", "Chemical_Structure_NPC", "ATC_1st_NPC", "ATC_2nd_NPC", "ATC_3rd_NPC", "Protein_RW", "Pathway_RW", "ADE_RW", "Indication_RW", "Chemical_Structure_RW", "ATC_1st_RW", "ATC_2nd_RW", "ATC_3rd_RW", "Protein_SRW", "Pathway_SRW", "ADE_SRW", "Indication_SRW", "Chemical_Structure_SRW", "ATC_1st_SRW", "ATC_2nd_SRW", "ATC_3rd_SRW", "Label"])
    for key1, value1 in drugs_dict.items():
        for key2, value2 in drugs_dict.items():
            if key1!=key2 and value1>value2:
                writer.writerow([key1, key2, drug_protein_meta_path[value1][value2] , drug_pathway_meta_path[value1][value2], drug_ade_meta_path[value1][value2], drug_ind_meta_path[value1][value2], drug_chem_struct_meta_path[value1][value2], drug_atc_1st_meta_path[value1][value2],  drug_atc_2nd_meta_path[value1][value2],  drug_atc_3rd_meta_path[value1][value2], drug_protein_NPC[value1][value2], drug_pathway_NPC[value1][value2], drug_ade_NPC[value1][value2], drug_ind_NPC[value1][value2], drug_chem_NPC[value1][value2], drug_atc_1st_NPC[value1][value2],  drug_atc_2nd_NPC[value1][value2],  drug_atc_3rd_NPC[value1][value2], drug_protein_RW[value1][value2] , drug_pathway_RW[value1][value2], drug_ade_RW[value1][value2], drug_ind_RW[value1][value2], drug_chem_RW[value1][value2], drug_atc_1st_RW[value1][value2],  drug_atc_2nd_RW[value1][value2],  drug_atc_3rd_RW[value1][value2], drug_protein_SRW[value1][value2], drug_pathway_SRW[value1][value2], drug_ade_SRW[value1][value2], drug_ind_SRW[value1][value2], drug_chem_SRW[value1][value2], drug_atc_1st_SRW[value1][value2],  drug_atc_2nd_SRW[value1][value2],  drug_atc_3rd_SRW[value1][value2], Labelled_data[value1][value2] ])


print('It is done! Congrats')