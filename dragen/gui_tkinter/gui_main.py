import sys
import os
from tkinter import ttk, messagebox, Toplevel, Tk, DoubleVar, Spinbox, Label, Button, W, E, Text
from tkinter import Checkbutton, BooleanVar, IntVar, Radiobutton, END
from tkinter.filedialog import askopenfilename, askdirectory
from PIL import ImageTk, Image
import multiprocessing
from dragen.main2D import DataTask2D
from dragen.main3D import DataTask3D


def open_file1():
    global name1
    name1 = askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if name1:
        phase_1_text.delete("1.0", END)
        phase_1_text.insert(END, name1)


def open_file2():
    global name2
    name2 = askopenfilename(title="Select file", filetypes=(("CSV Files", "*.csv"),))
    if name2:
        phase_2_text.delete("1.0", END)
        phase_2_text.insert(END, name2)

def get_root():
    global root_dir
    root_dir = askdirectory()
    if root_dir:
        root_dir_text.delete("1.0", END)
        root_dir_text.insert(END, root_dir)


# initialize main_window
root = Tk()
root.title("DRAGen - RVE Generator Tool")
root.geometry('900x700')  # (width x length)
root.resizable(height=False, width=False)

# set Logo
exe = sys.argv[0][-3:]
exe_flag = False
if exe == 'exe':
    exe_flag = True
    cwd = os.path.abspath(os.getcwd())
    if cwd[-4:] == 'dist':
        img_path = cwd + '/../dragen/gui_tkinter'
    else:
        img_path = cwd + '/dragen/gui_tkinter'
else:
    img_path = '.'
img = Image.open(img_path + '/Logo.png')
width, height = img.size
img = img.resize((width//3, height//3))
logo = ImageTk.PhotoImage(img)
panel = Label(root, image=logo)
panel.grid(row=1, column=5, columnspan=3, rowspan=20, sticky=E, padx=10, pady=3)

# Define Default Values for variables box_size, npts, packingratio, etc...
var = DoubleVar(value=22)  # initial value
box_size_input = Spinbox(root, from_=10, to=100, width=10, textvariable=var)
var = DoubleVar(value=22) # initial value
points_input = Spinbox(root, from_=10, to=512, width=10, textvariable=var)
var = DoubleVar(value=0.5)  # initial value
pack_ratio_input = Spinbox(root, from_=0.2, to=0.7, width=10, increment=.1, textvariable=var)
bands_input = Spinbox(root, from_=0, to=100, width=10)
var = DoubleVar(value=3)  # initial value
bandwidth_input = Spinbox(root, from_=1, to=100, width=10, increment=.1, textvariable=var)

# csv-Buttons
first_input_file = Button(root, text='Phase 1', command=open_file1)
second_input_file = Button(root, text='Phase 2', command=open_file2)
root_dir_btn = Button(root, text='save files', command=get_root)

var = DoubleVar(value=1)  # initial value
number_of_rve_input = Spinbox(root, from_=1, to=100, width=10, textvariable=var)

# Visualization checkbox
gen_visuals = BooleanVar()
gen_visuals.set(False)
visual_cbox = Checkbutton(root, text ="Visualization", variable = gen_visuals)

# Dimensionality checkbox Default 3D
dimension_input = IntVar()
dimension_input.set(3)
twoD_cbox = Radiobutton(root, text="2D RVE", variable=dimension_input, value=2)
thrD_cbox = Radiobutton(root, text="3D RVE", variable=dimension_input, value=3)

# Set Default Text for filepath text boxes
phase_1_text = Text(root, height=1, width=40)
phase_1_text.grid(row=16, column=1, columnspan=4, sticky=W, padx=3, pady=3)
phase_1_text.insert(END, 'Chose a csv-file for the first phase!')

phase_2_text = Text(root, height=1, width=40)
phase_2_text.grid(row=20, column=1, columnspan=4, sticky=W, padx=3, pady=3)
phase_2_text.insert(END, 'Chose a csv-file for the second phase!')

root_dir_text = Text(root, height=1, width=40)
root_dir_text.grid(row=24, column=1, columnspan=4, sticky=W, padx=3, pady=3)
root_dir_text.insert(END, 'Chose a directory to store the RVEs!')


def rveGeneration(file1, file2):
    last_RVE = int(number_of_rve_input.get())
    progress = 0
    dim = int(dimension_input.get())
    n = Label(root, text="RVE generation in progress... 0%").grid(row=26, column=1, sticky=W, padx=3, pady=3)

    if dimension_input.get() == 2:
        obj = DataTask2D(box_size=int(box_size_input.get()), n_pts=int(points_input.get()),
                         number_of_bands=int(bands_input.get()), bandwidth=float(bandwidth_input.get()),
                         shrink_factor=float(pack_ratio_input.get()), file1=file1, file2=file2,
                         store_path=str(root_dir), gui_flag=True, anim_flag=gen_visuals.get(), exe_flag=exe_flag)
        grains_df = obj.initializations(dimension=dim)

    elif dimension_input.get() == 3:
        obj = DataTask3D(box_size=int(box_size_input.get()), n_pts=int(points_input.get()),
                         number_of_bands=int(bands_input.get()), bandwidth=float(bandwidth_input.get()),
                         shrink_factor=float(pack_ratio_input.get()), file1=file1, file2=file2,
                         store_path=str(root_dir), gui_flag=True, anim_flag=gen_visuals.get(), exe_flag=exe_flag)
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

        a = Label(root, text="Boxsize: ", ).grid(row=0, column=0, sticky=W, padx=5, pady=5)
        b = Label(root, text="Number of points on edge: ").grid(row=2, column=0, sticky=W, padx=5, pady=5)
        c = Label(root, text="Packing ratio: ").grid(row=4, column=0, sticky=W, padx=5, pady=5)
        d = Label(root, text="Number of bands: ").grid(row=6, column=0, sticky=W, padx=5, pady=5)
        e = Label(root, text="Bandwidth: ").grid(row=8, column=0, sticky=W, padx=5, pady=5)
        f = Label(root, text="Check for the desired RVE dimension: ").grid(row=10, column=0, sticky=W, padx=5, pady=5)
        o = Label(root, text="Number of RVEs required: ").grid(row=12, column=0, sticky=W, padx=5, pady=5)
        g = Label(root, text="Upload first input file*: ").grid(row=14, column=0, sticky=W, padx=5, pady=5)
        h = Label(root, text="Upload second input file: ").grid(row=18, column=0, sticky=W, padx=5, pady=5)
        i = Label(root, text="Set Workdirectory: ").grid(row=22, column=0, sticky=W, padx=5, pady=5)

        l = Label(root, text="* Required").grid(row=24, column=0, sticky=W, padx=5, pady=5)

        box_size_input.grid(row=0, column=1, sticky=W, padx=5, pady=5)
        points_input.grid(row=2, column=1, sticky=W, padx=5, pady=5)
        pack_ratio_input.grid(row=4, column=1, sticky=W, padx=5, pady=5)
        bands_input.grid(row=6, column=1, sticky=W, padx=5, pady=5)
        bandwidth_input.grid(row=8, column=1, sticky=W, padx=5, pady=5)
        twoD_cbox.grid(row=10, column=2, sticky=W, padx=5, pady=5)
        thrD_cbox.grid(row=10, column=1, sticky=W, padx=5, pady=5)
        number_of_rve_input.grid(row=12, column=1, sticky=W, padx=5, pady=5)
        first_input_file.grid(row=14, column=1, sticky=W, padx=5, pady=5)
        second_input_file.grid(row=18, column=1, sticky=W, padx=5, pady=5)
        visual_cbox.grid(row=19, column=0, sticky=W, padx=5, pady=5)
        root_dir_btn.grid(row=22, column=1, sticky=W, padx=5, pady=5)

        self.submit_btn = Button(root, text="Submit", command=self.validateData)
        self.submit_btn.grid(row=25, column=1, padx=5, pady=5)

        self.cancel_btn = Button(root, text="Cancel", command=self.terminateProcess)
        self.cancel_btn.grid(row=25, column=2, padx=5, pady=5)

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
