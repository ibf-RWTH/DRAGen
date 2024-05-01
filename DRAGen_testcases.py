import os
#import sys

def run_test_cases(test_folder):
    files = os.listdir(test_folder)
    
    for file in files:
        try:
            #with open('output.txt', 'a') as output_file:
                if file.endswith(".py"):
                    file_path = os.path.join(test_folder, file)
                    module_name = file[:-3]
                    test_module = __import__(f"Test_Cases.{module_name}", fromlist=[module_name])


                    print("Test case", module_name, "successful.")
                    #sys.stdout = open('output.txt','w')
                    #with open('output.txt', 'a') as output_file:
                    #    with open('output.txt','a') as stdout_file:
                    #        original_stdout = sys.stdout
                    #
                    #        sys.stdout = stdout_file
                    #        try:
                    #            pass
                    #        finally:
                    #            sys.stdout = original_stdout

            #print("Test case", module_name, "successful.")

        except Exception as e:
            print(f"An error occurred while running test case {module_name}:", e)
            print("Test case", module_name, "failed.")

if __name__ == "__main__":
    test_folder = "Test_Cases"
    run_test_cases(test_folder)
