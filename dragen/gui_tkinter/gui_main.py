import sys
from tkinter import ttk, messagebox, Toplevel, Tk, DoubleVar, Spinbox, Label, Button, W
import multiprocessing
from tkinter.filedialog import askopenfilename

from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D


def open_file1():
    global name1
    name1 = askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if name1:
        i = Label(root, text=name1).grid(row=16, column=1, sticky=W, padx=3, pady=3)


def open_file2():
    global name2
    name2 = askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if name2:
        j = Label(root, text=name2).grid(row=20, column=1, sticky=W, padx=3, pady=3)


root = Tk()
root.title("DRAGen - a RVE Generator Tool")
root.geometry('550x480')  # (width x length)

var = DoubleVar(value=22)  # initial value
box_size_input = Spinbox(root, from_=10, to=100, width=10, textvariable=var)
var = DoubleVar(value=22)
points_input = Spinbox(root, from_=10, to=512, width=10, textvariable=var)
var = DoubleVar(value=0.5)  # initial value
pack_ratio_input = Spinbox(root, from_=0.2, to=0.7, width=10, increment=.1, textvariable=var)
bands_input = Spinbox(root, from_=0, to=100, width=10)
var = DoubleVar(value=3)  # initial value
bandwidth_input = Spinbox(root, from_=1, to=100, width=10, increment=.1, textvariable=var)
dimension_input = Spinbox(root, from_=2, to=3, width=10)
first_input_file = Button(root, text='Upload 1', command=open_file1)
second_input_file = Button(root, text='Upload 2', command=open_file2)
var = DoubleVar(value=1)  # initial value
number_of_rve_input = Spinbox(root, from_=1, to=100, width=10, textvariable=var)


def rveGeneration(file1, file2):
    last_RVE = int(number_of_rve_input.get())
    progress = 0
    dim = int(dimension_input.get())
    n = Label(root, text="RVE generation in progress... 0%").grid(row=26, column=1, sticky=W, padx=3, pady=3)

    if dim == 2:
        obj = DataTask2D(int(box_size_input.get()), int(points_input.get()), int(bands_input.get()),
                       float(bandwidth_input.get()),  float(pack_ratio_input.get()), file1, file2, True)
        grains_df = obj.initializations(dimension=dim)
    elif dim == 3:
        obj = DataTask3D(int(box_size_input.get()), int(points_input.get()), int(bands_input.get()),
                         float(bandwidth_input.get()), float(pack_ratio_input.get()),
                         file1, file2, True)

        grains_df = obj.initializations(dimension=dim)
    for i in range(last_RVE):
        progress_label = "RVE generation in progress... {}%".format(progress)
        n = Label(root, text=progress_label).grid(row=26, column=1, sticky=W, padx=3, pady=3)
        obj.rve_generation(i, grains_df)
        progress = progress + 100/(last_RVE)

    m = Label(root, text="RVE generation completed successfully!!!").grid(row=26, column=1, sticky=W, padx=3, pady=3)


class ProcessWindow(Toplevel):
    def __init__(self, parent, process):
        Toplevel.__init__(self, parent)
        self.parent = parent
        self.process = process
        #terminate_button = ttk.Button(self, text="Cancel", command=self.cancel)
        #terminate_button.grid(row=0, column=0)
        self.grab_set()  # so you can't push submit multiple times

    def cancel(self):
        self.process.terminate()
        self.destroy()
        messagebox.showinfo(message='Process has been terminated!!!', title="RVE Generation Canceled")

    def launch(self):
        self.process.start()
        self.after(100, self.isAlive)  # Starting the loop to check when the process is going to end

    def isAlive(self):
        if self.process.is_alive():  # Process still running
            self.after(100, self.isAlive)  # Going again...
        elif self:
            # Process finished
            messagebox.showinfo(message="Successful Run!!!", title="RVE Generation Completed")
            self.destroy()


class MainApplication(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        a = Label(root, text="Boxsize: ", anchor='w').grid(row=0, column=0, sticky=W, padx=5, pady=5)
        b = Label(root, text="Number of points on edge: ").grid(row=2, column=0, sticky=W, padx=5, pady=5)
        c = Label(root, text="Packing ratio: ").grid(row=4, column=0, sticky=W, padx=5, pady=5)
        d = Label(root, text="Number of bands: ").grid(row=6, column=0, sticky=W, padx=5, pady=5)
        e = Label(root, text="Bandwidth: ").grid(row=8, column=0, sticky=W, padx=5, pady=5)
        f = Label(root, text="Type 2 for 2D and 3 for 3D RVE: ").grid(row=10, column=0, sticky=W, padx=5, pady=5)
        o = Label(root, text="Number of RVEs required: ").grid(row=12, column=0, sticky=W, padx=5, pady=5)
        g = Label(root, text="Upload first input file*: ").grid(row=14, column=0, sticky=W, padx=5, pady=5)
        h = Label(root, text="Upload second input file: ").grid(row=18, column=0, sticky=W, padx=5, pady=5)
        k = Label(root, text="* Required").grid(row=24, column=0, sticky=W, padx=5, pady=5)

        box_size_input.grid(row=0, column=1)
        points_input.grid(row=2, column=1)
        pack_ratio_input.grid(row=4, column=1)
        bands_input.grid(row=6, column=1)
        bandwidth_input.grid(row=8, column=1)
        dimension_input.grid(row=10, column=1)
        number_of_rve_input.grid(row=12, column=1)
        first_input_file.grid(row=14, column=1)
        second_input_file.grid(row=18, column=1)

        self.submit_btn = Button(root, text="Submit", command=self.validateData)
        self.submit_btn.grid(row=22, column=0, padx=5, pady=5)

        self.cancel_btn = Button(root, text="Cancel", command=self.terminateProcess)
        self.cancel_btn.grid(row=22, column=1, padx=5, pady=5)

    def validateData(self):
        if int(points_input.get()) % 2 == 0 and name1:
            proc = multiprocessing.Process(target=rveGeneration(name1, name2))
            process_window = ProcessWindow(self, proc)
            process_window.launch()
        else:
            self.validationError()

    def terminateProcess(self):
        sys.exit()

    def validationError(self):
        popup = Tk()
        popup.title("Error!")
        label = Label(popup, text="Make sure that the 'Number of Points on Edge' input is even and there is a CSV file uploaded.")
        label.pack(side="top", fill="x", pady=10)
        popup.mainloop()


def main():
    my_app = MainApplication(root, padding=4)
    my_app.grid(column=0, row=0)
    root.mainloop()


if __name__ == '__main__':
    # create global variables
    name1 = None
    name2 = None
    main()
