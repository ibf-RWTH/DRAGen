from tkinter import *
from tkinter.ttk import Progressbar

from dragen.main import DataTask

window = Tk()
window.title("DRAGen - a RVE Generator Tool")
window.geometry('350x250')

# Form
a = Label(window, text="Boxsize: ", anchor='w').grid(row=0, column=0)
b = Label(window, text="Number of points on edge: ").grid(row=2, column=0)
c = Label(window, text="Packing ratio: ").grid(row=4, column=0)
d = Label(window, text="Number of bands: ").grid(row=6, column=0)
e = Label(window, text="Bandwidth: ").grid(row=8, column=0)
f = Label(window, text="Growing speed for grain growth: ").grid(row=10, column=0)

var = DoubleVar(value=50)  # initial value
box_size_input = Spinbox(window, from_=0, to=100, width=10, textvariable=var)
box_size_input.grid(row=0, column=1)

var = DoubleVar(value=50)
points_input = Spinbox(window, from_=10, to=512, width=10, textvariable=var)
points_input.grid(row=2, column=1)

var = DoubleVar(value=0.5)  # initial value
pack_ratio_input = Spinbox(window, from_=0.2, to=0.7, width=10, increment=.1, textvariable=var)
pack_ratio_input.grid(row=4, column=1)

bands_input = Spinbox(window, from_=0, to=100, width=10)
bands_input.grid(row=6, column=1)

var = DoubleVar(value=3)  # initial value
bandwidth_input = Spinbox(window, from_=1, to=15, width=10, increment=.1, textvariable=var)
bandwidth_input.grid(row=8, column=1)

speed_input = Spinbox(window, from_=1, to=10, width=10)
speed_input.grid(row=10, column=1)


def generateRVEs():
    popup = Tk()
    popup.title("RVE generation in progress...")
    progress = Progressbar(popup, orient=HORIZONTAL, length=100, mode='determinate')
    progress.pack(pady=10)
    last_RVE = 3
    progress_value = 0
    obj = DataTask(int(box_size_input.get()), int(points_input.get()), int(bands_input.get()),
                   float(bandwidth_input.get()), int(speed_input.get()))
    convert_list, phase1, phase2 = obj.initializations()
    for i in range(last_RVE + 1):
        obj.rve_generation(i, convert_list, phase1, phase2)
        progress_value += 100/(last_RVE+1)
        progress['value'] = progress_value

    popup.mainloop()


def validationError():
    popup = Tk()
    popup.title("Error!")
    label = Label(popup, text="Number of points on edge input has to be even.")
    label.pack(side="top", fill="x", pady=10)
    popup.mainloop()


def validateData():
    if int(points_input.get()) % 2 == 0:
        generateRVEs()

    else:
        validationError()


submit_btn = Button(window, text="Submit", command=validateData).grid(row=14, column=0)
cancel_btn = Button(window, text="Cancel").grid(row=14, column=1)
window.mainloop()
