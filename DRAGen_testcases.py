import os

def run_test_cases(test_folder):
    files = os.listdir(test_folder)
    
    for file in files:
        try:
            # to check if its a Python file
            if file.endswith(".py"):
                file_path = os.path.join(test_folder, file)
                
                module_name = file[:-3]
                
                test_module = __import__(f"Test_Cases.{module_name}", fromlist=[module_name])
                
                if hasattr(test_module, "run"):
                    print("Running test case:", module_name)
                    print('the cwd is: ', os.getcwd())
                    test_module.run()
                    print("Test case", module_name, "successful.")
                else:
                    print(f"Test case {module_name} does not have a 'run' function.")
        except Exception as e:
            print(f"An error occurred while running test case {module_name}:", e)
            print("Test case", module_name, "failed.")

if __name__ == "__main__":
    test_folder = "Test_Cases"
    run_test_cases(test_folder)