import tkinter as tk
from tkinter import messagebox
import sys
import sys
from tkinter import *
from tkinter import filedialog
import glob
from os.path import dirname, basename, isfile, join
import itertools


def train_or_predict():
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    root.title("Train_or_predict")
    root.protocol("WM_DELETE_WINDOW", on_closing)
    v = tk.BooleanVar()
    w = tk.BooleanVar()
    tk.Label(root, text="""Do you want to train a new model,
             or use an existing one to predict results?""",
             justify=tk.LEFT, padx=20).grid(row=0, column=0)
    tk.Radiobutton(root, text="Create", indicatoron=0,
                   width=30, padx=20, variable=v, value=1,
                   command=root.destroy).grid(row=2, column=0)
    tk.Radiobutton(root, text="Update", indicatoron=0,
                   width=30, padx=20, variable=w, value=1,
                   command=root.destroy).grid(row=3, column=0)
    root.mainloop()
    return v.get()


def get_models():
    modules = glob.glob(join(dirname(__file__), "../Models/*.py"))
    classifiers = ([basename(f)[:-3] for f in modules
                    if isfile(f) and not
                    (f.endswith("__init__.py") or f.endswith("classifier.py"))
                    ])
    box_var = []
    boxes = []
    box_num = 0
    root = tk.Tk()

    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
            sys.exit('Quitting...')

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.geometry("450x300+120+120")
    tk.Label(
        root, text="Select the classifiers you want to tune and train",
        justify=tk.LEFT, font=("Arial", 14), padx=5, pady=10
        ).grid(row=0, column=0, columnspan=2)
    r = 0
    for clf in classifiers[:int(len(classifiers)/2) + 1]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(1)
        boxes[box_num].grid(row=r + 1, column=0)
        box_num += 1
        r += 1
    confirm_row = r + 1
    r = 0
    for clf in classifiers[int(len(classifiers)/2) + 1:]:
        box_var.append(tk.IntVar())
        boxes.append(
            tk.Checkbutton(
                root, text=clf,
                variable=box_var[box_num]
                )
            )
        box_var[box_num].set(1)
        boxes[box_num].grid(row=r + 1, column=1)
        box_num += 1
        r += 1

    tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
              command=root.destroy, justify=tk.CENTER
              ).grid(row=confirm_row, column=0, pady=10, columnspan=2)
    root.mainloop()
    mask = [val.get() for val in box_var]
    return list(itertools.compress(classifiers, mask))


def train():

    classifiers = get_models()
    if classifiers:
        modules = glob.glob(join(dirname(__file__), "../Models/*.joblib"))
        models = ([basename(f).replace('_model.joblib', '')
                   for f in modules if isfile(f)])
        common_clf = list(set(classifiers).intersection(models))
        diff_clf = list(set(classifiers).difference(models))
        box_var = []
        boxes = []
        box_num = 0
        root = tk.Tk()

        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                sys.exit('Quitting...')

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.geometry("450x300+120+120")
        tk.Label(
            root, text=("The following classifiers have already"
                        + "been trained.\n"
                        + "Do you want to overwrite any of them?"),
            justify=tk.LEFT, font=("Arial", 14), padx=5, pady=10
                ).grid(row=0, column=0, columnspan=2)
        r = 0
        for clf in common_clf[:int(len(common_clf)/2) + 1]:
            box_var.append(tk.IntVar())
            boxes.append(
                tk.Checkbutton(
                    root, text=clf,
                    variable=box_var[box_num]
                    )
                )
            box_var[box_num].set(0)
            boxes[box_num].grid(row=r + 1, column=0)
            box_num += 1
            r += 1
        confirm_row = r + 1
        r = 0
        for clf in common_clf[int(len(common_clf)/2) + 1:]:
            box_var.append(tk.IntVar())
            boxes.append(
                tk.Checkbutton(
                    root, text=clf,
                    variable=box_var[box_num]
                    )
                )
            box_var[box_num].set(0)
            boxes[box_num].grid(row=r + 1, column=1)
            box_num += 1
            r += 1

        tk.Button(root, text="Confirm", width=10, relief=tk.RAISED,
                  command=root.destroy, justify=tk.CENTER
                  ).grid(row=confirm_row, column=0, pady=10, columnspan=2)
        root.mainloop()
        mask = [val.get() for val in box_var]
        return list(diff_clf) + list(itertools.compress(common_clf, mask))
    else:

        def countdown(count):
            # change text in label
            label['text'] = f"No model selected, exiting program in {count}"
            if count > 0:
                # call countdown again after 1000ms (1s)
                root.after(1000, countdown, count-1)
            else:
                close()

        def close():
            root.destroy()

        root = tk.Tk()
        root.geometry("450x300+120+120")
        count = 5
        label = tk.Label(root,
                         text=f"No model selected, exiting program in {count}")
        label.place(x=35, y=15)

        # call countdown first time
        countdown(count)
        root.mainloop()

# def predict():

#     root = Tk()
#     root.filename = filedialog.askopenfilename(
#                         text='Select a file', initialdir="/",
#                         title="Select file",
#                         filetypes=(("csv", "*.csv"), ("all files", "*.*"))
#                         )
#     root.mainloop()
#     print(root.filename)
