from fileinput import filename
import numpy as np
import pathlib
from tqdm import tqdm
import os
from pathlib import Path

global_path = str(pathlib.Path(__file__).parent.parent.resolve())
print(global_path)
data = "style_category_dataset"
#target_folder = f"examples/language_model/{data}/"
target_folder = f"{global_path}/examples/language_model/{data}/"
print(target_folder)
chunk_size = 5000 


def generate_category_locality_matrix(split_name, chunk_size):

    filename = f'{target_folder}{split_name}train.txt.category.npy'
    file_path = Path(filename)

    try:
        os.remove(filename)
        print("Deleted Previous File")
    except OSError as e:
        print(f"No existing files found: {e}")
        pass

    print(split_name)
    test_sections = []
    testtrain_sections = []
    locality = []

    with open(f'{target_folder}{split_name}.txt.category') as test_section_file, \
            open(f'{target_folder}{split_name}train.txt.category') as testtrain_section_file:
        for line in test_section_file:
            test_sections.append(line.strip())
        for line in testtrain_section_file:
            testtrain_sections.append(line.strip())
    
    print("\n")
    print(f"Test Domains:{len(test_sections)}")
    print(f"TestTrain Domains:{len(testtrain_sections)}")
    print("\n") 
    
    # Create nmap output
    output_array = np.memmap(filename, dtype='int8', mode='w+', shape=(len(test_sections),len(testtrain_sections)))
    print(output_array.shape)
    return True

    i = 0
    for p in tqdm(test_sections):
            
        temp_loc = []
        for t in testtrain_sections:
            if p == t:
                temp_loc.append(1)
            else:
                temp_loc.append(0)

        output_array[i,:] = temp_loc
        i+=1
        del temp_loc
        
        if i % chunk_size  == 0 :
            tqdm.write(f"Flush:")
            output_array.flush()

    output_array.flush()

def generate_style_locality_matrix(split_name, chunk_size):

    filename = f'{target_folder}{split_name}train.txt.style.npy'
    file_path = Path(filename)

    try:
        os.remove(filename)
        print("Deleted Previous File")
    except OSError as e:
        print(f"No existing files found: {e}")
        pass

    print(split_name)
    test_sections = []
    testtrain_sections = []
    locality = []

    with open(f'{target_folder}{split_name}.txt.style') as test_section_file, \
            open(f'{target_folder}{split_name}train.txt.style') as testtrain_section_file:
        for line in test_section_file:
            test_sections.append(line.strip())
        for line in testtrain_section_file:
            testtrain_sections.append(line.strip())
    
    print("\n")
    print(f"Test Domains:{len(test_sections)}")
    print(f"TestTrain Domains:{len(testtrain_sections)}")
    print("\n") 
    
    # Create nmap output
    output_array = np.memmap(filename, dtype='int8', mode='w+', shape=(len(test_sections),len(testtrain_sections)))
    print(output_array.shape)

    i = 0
    for p in tqdm(test_sections):
            
        temp_loc = []
        for t in testtrain_sections:
            if p == t:
                temp_loc.append(1)
            else:
                temp_loc.append(0)

        output_array[i,:] = temp_loc
        i+=1
        del temp_loc
        
        if i % chunk_size  == 0 :
            tqdm.write(f"Flush:")
            output_array.flush()

    output_array.flush()


generate_category_locality_matrix('test', chunk_size)
generate_category_locality_matrix('valid', chunk_size)

generate_style_locality_matrix('test', chunk_size)
generate_style_locality_matrix('valid', chunk_size)

filename = f'{target_folder}{"valid"}train.txt.style.npy'

output_array = np.memmap(filename, dtype='int8', mode='r', shape=(96195,384775))

#Shape Valid = (58905, 392700)

#Shape Test = (69300, 403059)