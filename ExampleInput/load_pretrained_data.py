

import ExampleInput

def main():
    pretrained_data_paths = [
        './ExampleInput/Ferrite/TrainedData_Ferrite.pkl',
        './ExampleInput/Martensite/TrainedData_Martensite.pkl',
        './ExampleInput/Pearlite/TrainedData_Pearlite.pkl',
        './ExampleInput/Bainite/TrainedData_Bainite.pkl',
        './ExampleInput/Austenite/TrainedData_Austenite.pkl',
        './ExampleInput/Inclusions/TrainedData_Inclusion.pkl',
        './ExampleInput/Banding/TrainedData_Band.csv',
        './ExampleInput/Substructure/example_pag_inp.csv'
    ]  
    
    for path in pretrained_data_paths:
        loaded_data = ExampleInput.load_pretrained_data(path)
        
        if loaded_data is not None:
            print(f"Pretrained data loaded successfully from {path}:\n{loaded_data.head()}")
        else:
            print(f"Failed to load pretrained data from {path}.")

if __name__ == "__main__":
    main()