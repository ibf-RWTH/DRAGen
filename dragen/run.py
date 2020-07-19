from dragen.main import DataTask

if __name__ == "__main__":
    obj = DataTask()
    try:
        obj.run_main()
    except KeyboardInterrupt:
        print('Exiting!!!')
