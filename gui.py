import os
import sys
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from perceptron import perceptron

def str2float(strlist):
    return [round(float(i)) if float(i).is_integer() else float(i) for i in strlist]



class gui():
    def __init__(self, app_name, app_width, app_height):
        self.x_train_without_x0 = []
        self.x_test_without_x0 = []
        self.y_train = []
        self.y_test = []
        self.stop_animation_flag = True

        self.file_name = ''
        self.data = None
        self.inputs = np.array([])
        self.targets = np.array([])
        self.train_result = []
        self.test_result = []

        self.epoch = 0
        self.lr = 0
        self.dim = 0
        self.epoch_result = 1
        self.best_epoch = None

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white', padx=10, pady=10)
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(str(app_width) + 'x' + str(app_height))


        # components initialization
        self.graph_frame = tk.Frame(self.container, width=1320, height=420, bg='white')
        self.setting_frame = tk.Frame(self.container, width=500, height=320, bg='white')
        self.result_frame = tk.Frame(self.container, width=850, height=320, bg='white')

        self.canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.canvas.get_tk_widget().config(width=430, height=400)
        self.train_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.train_canvas.get_tk_widget().config(width=430, height=400)
        self.test_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.test_canvas.get_tk_widget().config(width=430, height=400)
        self.train_result_canvas = FigureCanvasTkAgg(master = self.result_frame)
        self.train_result_canvas.get_tk_widget().config(width=500, height=310)

        self.setting_text = tk.Label(self.setting_frame, text='Settings', bg='white')
        self.file_box = tk.Label(self.setting_frame, text='Current File: No File Selected', bg='white', wraplength=300)
        self.load_btn = tk.Button(master = self.setting_frame,  
                     command = self.show_local, 
                     height = 1,  
                     width = 10, 
                     text = "Load from Local",
                     highlightbackground='white') 
        self.select_btn = tk.Button(master = self.setting_frame,
                        command = self.show,
                        height = 1,
                        width = 10,
                        text = "Built-in dataset",
                        highlightbackground='white')
        
        self.dim_label = tk.Label(self.setting_frame, text='Dim: ', bg='white')
        self.dim_text = tk.Label(self.setting_frame, text='', bg='white')
        self.sample_num_label = tk.Label(self.setting_frame, text='Numbers of Samples: ', bg='white')
        self.sample_num = tk.Label(self.setting_frame, text='', bg='white')
        self.epoch_label = tk.Label(self.setting_frame, text='Epoch:', bg='white')
        self.epoch_box = tk.Spinbox(self.setting_frame, increment=1, from_=0, width=5, bg='white', textvariable=tk.StringVar(value='100'))

        self.lrn_rate_label = tk.Label(self.setting_frame, text='Learning Rate:', bg='white')
        self.lrn_rate_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.01, from_=0.0,to=1, width=5, bg='white', textvariable=tk.StringVar(value='0.01'))
        self.train_btn = tk.Button(master = self.setting_frame,  
                     command = self.train, 
                     height = 2,  
                     width = 10, 
                     text = "Train Data",
                     highlightbackground='white') 
        self.save_btn = tk.Button(master = self.setting_frame,  
                     command = self.save, 
                     height = 2,  
                     width = 16, 
                     text = "Save Full Data Graph",
                     highlightbackground='white')
        self.train_sample_label = tk.Label(master = self.setting_frame, text='Number of Training Data', bg='white')
        self.train_sample_num = tk.Label(master = self.setting_frame, text='', bg='white')
        self.test_sample_label = tk.Label(master = self.setting_frame, text='Number of Testing Data', bg='white')
        self.test_sample_num = tk.Label(master = self.setting_frame, text='', bg='white')
        
        
        self.current_result = tk.Label(self.result_frame, text='Currently Showing result of epoch  ', bg='white')
        self.current_epoch = tk.Spinbox(self.result_frame, increment=1, from_=1, to=int(self.epoch_box.get()), width=5, bg='white', textvariable=tk.StringVar(value='1'), command=self.update_graph)
        self.show_update_btn = tk.Button(master = self.result_frame,  
                     command = self.update_graph, 
                     height = 1,  
                     width = 5, 
                     text = "Show",
                     highlightbackground='white')
        self.stop_update_btn = tk.Button(master = self.result_frame,
                        command=self.show_last_result,
                        height = 1,
                        width = 5,
                        text = "Stop",
                        highlightbackground='white')
        self.result_text = tk.Label(self.result_frame, text='Results', bg='white')
        self.train_acc_label = tk.Label(self.result_frame, text='Train Accuracy: ', bg='white')
        self.train_acc = tk.Label(self.result_frame, text='...', bg='white')
        self.test_acc_label = tk.Label(self.result_frame, text='Test Accuracy: ', bg='white')
        self.test_acc = tk.Label(self.result_frame, text='...', bg='white')
        self.weight_label = tk.Label(self.result_frame, text='Weight(%.2f): ', bg='white')
        self.weight = tk.Label(self.result_frame, text='...', bg='white')

        self.best_weight_label = tk.Label(self.result_frame, text='Best Weight(%.2f): ', bg='white')
        self.best_weight = tk.Label(self.result_frame, text='...', bg='white')
        self.best_acc_label = tk.Label(self.result_frame, text='Best Testing Accuracy: ', bg='white', wraplength=100, justify='left')
        self.best_acc = tk.Label(self.result_frame, text='...', bg='white')
        self.best_epoch_label = tk.Label(self.result_frame, text='Best Epoch: ', bg='white')
        self.best_epoch = tk.Label(self.result_frame, text='...', bg='white')

        # components placing
        self.setting_frame.place(x=5, y=5)
        self.result_frame.place(x=480, y=5)
        self.graph_frame.place(x=0, y=330)
        self.canvas.get_tk_widget().place(x=0, y=10)
        self.train_canvas.get_tk_widget().place(x=450, y=10)
        self.test_canvas.get_tk_widget().place(x=890, y=10)
        self.train_result_canvas.get_tk_widget().place(x=340, y=5)


        self.figure = None
        self.load_btn.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.select_btn.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.file_box.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.dim_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.dim_text.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.sample_num_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.sample_num.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.epoch_label.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.epoch_box.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.lrn_rate_label.grid(row=5, column=0, padx=5, pady=5, sticky='w')
        self.lrn_rate_box.grid(row=5, column=1, padx=5, pady=5, sticky='w')
        self.train_btn.grid(row=6, column=0, padx=5, pady=5, sticky='w')
        self.save_btn.grid(row=6, column=1, padx=5, pady=5, sticky='w')
        self.train_sample_label.grid(row=7, column=0, padx=5, pady=5, sticky='w')
        self.train_sample_num.grid(row=7, column=1, padx=5, pady=5, sticky='w')
        self.test_sample_label.grid(row=8, column=0, padx=5, pady=5, sticky='w')
        self.test_sample_num.grid(row=8, column=1, padx=5, pady=5, sticky='w')
        

    def save(self):
        if self.figure == None:
            messagebox.showerror('showerror', 'No Image to Save')
            print('No Image to Save')
            return
        filename = asksaveasfilename(initialfile = 'Untitled.png',defaultextension=".png",filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")])
        self.figure.savefig(filename)

    def show_local(self):
        file = tk.filedialog.askopenfilename()
        self.file_name = file.split('/')[-1]
        data = self.read_file(file)
        if file == '' or data == None:
            file = 'No File Selected'
            messagebox.showerror('showerror', 'Selected File Invalid')
            print('Selected File Invalid')
            self.init_all_member()
            return
        self.load(data, file)

    def show(self):
        top = tk.Toplevel(self.container)
        top.title("Select a Dataset")
        # Listbox to display dataset names
        listbox = tk.Listbox(top, height=30, width=40)
        listbox.pack()

        # Add dataset files to the Listbox
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller uses a temporary folder named _MEIPASS to extract files
            datasets_folder = os.path.join(sys._MEIPASS, "data")
        else:
            # In development mode
            datasets_folder = os.path.join(os.path.abspath("."), "data")
        # Directory where datasets are stored

    
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller uses a temporary folder named _MEIPASS to extract files
            basic_path = os.path.join(sys._MEIPASS, "data/basic")
        else:
            # In development mode
            basic_path = os.path.join(os.path.abspath("."), "data/basic")

        basic_files = os.listdir(basic_path)
        listbox.insert(tk.END, '--- Basic Datasets ---')
        for dataset in basic_files:
            listbox.insert(tk.END, os.path.join('basic', dataset))
        
        # Add a separator
        listbox.insert(tk.END, '')

        # Add datasets from 'extra' subfolder next
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller uses a temporary folder named _MEIPASS to extract files
            extra_path = os.path.join(sys._MEIPASS, "data/extra")
        else:
            # In development mode
            extra_path = os.path.join(os.path.abspath("."), "data/extra")

        extra_files = os.listdir(extra_path)
        listbox.insert(tk.END, '--- Extra Datasets ---')
        for dataset in extra_files:
            listbox.insert(tk.END, os.path.join('extra', dataset))

        listbox.bind('<Double-1>', lambda event: self.load(self.read_file(os.path.join(datasets_folder, listbox.get(tk.ACTIVE))), listbox.get(tk.ACTIVE)))
        
    def load(self, data, file):
        # Extract inputs and targets
        self.inputs = np.array([item[0] for item in data])
        self.targets = np.array([item[1] for item in data])
        self.dim = len(data[0][0])
        sample_num = len(data)
        self.sample_num.config(text=sample_num)
        self.dim_text.config(text=self.dim)
        self.file_box.config(text='Current File: ' + file)
        self.visualize(self.inputs, self.targets, 'Complete Data', self.canvas, None)
        

    def visualize(self, inputs, targets, data_range, canvas, weight):
        inputs = np.array(inputs)
        targets = np.array(targets)

        # Determine the dimensionality of the input data
        self.figure = plt.Figure(figsize=(5, 5), dpi=80)

        # 建立2d或3d的subplot，超過3d的資料則跳出視窗告知不支援
        if self.dim == 2:
            ax = self.figure.add_subplot(111)
        elif self.dim == 3:
            ax = self.figure.add_subplot(111, projection='3d')
        else:
            self.clear_all_graph()
            messagebox.showerror('showerror', 'Only supports 2D or 3D data visualization.')
            raise ValueError("This function only supports 2D or 3D data visualization.")

        # Scatter plot with automatic color assignment
        if self.dim == 2:
            scatter = ax.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='viridis')
            ax.set(xlabel='x1', ylabel='x2')
        elif self.dim == 3:
            scatter = ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=targets, cmap='viridis')
            ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

        # Add a colorbar to show the mapping of colors to target values
        # Place the legend outside of the chart
        ax.legend(*scatter.legend_elements(), title='Class', loc='upper right' , bbox_to_anchor=(1, 0.5))

        ax.set_title('Data Visualization(' + data_range + '): ' + self.file_name)

        # If weight is provided, plot the decision boundary
        if weight is not None:
            w = np.array(weight)

            # Assuming inputs is your dataset, extract the min and max of the first feature (x1)
            if self.dim == 2:
                x1 = np.linspace(np.min(inputs[:, 0])-10, np.max(inputs[:, 0])+10, 100)
                # Calculate x2 using the perceptron decision boundary equation
                x2 = (w[0] - w[1] * x1) / w[2]
                # Plotting the decision boundary line
                ax.plot(x1, x2, label=f'{weight[1]}x1 + {weight[2]}x2 + {weight[0]} = 0')
                ax.set_xlim([np.min(inputs[:, 0]-1), np.max(inputs[:, 0])+1])

                ax.set_ylim([np.min(inputs[:, 1]-1), np.max(inputs[:, 1]+1)])
                ax.set_aspect('auto')

            elif self.dim == 3:
                x1 = np.linspace(np.min(inputs[:, 0]), np.max(inputs[:, 0]), 100)
                x2 = np.linspace(np.min(inputs[:, 1]), np.max(inputs[:, 1]), 100)
                x1, x2 = np.meshgrid(x1, x2)
                x3 = (weight[1] * x1 + weight[2] * x2 + weight[0]) / -weight[3]
                ax.plot_surface(x1, x2, x3, alpha=0.5)
                x3 = (weight[1] * x1 + weight[2] * x2 + weight[0]) / -weight[3]
                ax.plot_surface(x1, x2, x3, alpha=0.5)
               
            

        # Set figure to the tkinter-matplotlib canvas
        canvas.figure = self.figure
        canvas.draw()
        # Placing the canvas on graph_frame
        canvas.get_tk_widget().update()
    
    def train(self):
        

        if self.data == None:
            messagebox.showerror('showerror', 'No Data to Train')
            print('No Data to Train')
            return

        if self.lrn_validation() == False:
            return
        
        try:
            self.load_btn.config(state='disabled')
            self.select_btn.config(state='disabled')
            self.train_btn.config(state='disabled')
            self.save_btn.config(state='disabled')
            self.show_update_btn.config(state='disabled')
            self.stop_update_btn.place(x=180, y=30)
            print('Training...')
            print('Data:', self.data)
            print('Epoch:', self.epoch_box.get())
            print('Learning Rate:', self.lrn_rate_box.get())
            self.epoch = int(self.epoch_box.get())
            if(self.dim > 2):
                messagebox.showerror('showerror', "Can't train data with more than 2 dimensions")
                self.train_btn.config(state='normal')
                self.load_btn.config(state='normal')
                self.select_btn.config(state='normal')
                self.save_btn.config(state='normal')
                self.stop_update_btn.place_forget()
                return
            # Initialize perceptron
            p = perceptron()
            p.init_weight(self.dim)

            X_train, x_test, Y_train, y_test = p.split_data(self.data)
            X_train_without_x0 = [ i[1:] for i in X_train]
            x_test_without_x0 = [ i[1:] for i in x_test]
            self.x_train_without_x0 = X_train_without_x0
            self.x_test_without_x0 = x_test_without_x0
            self.y_train = Y_train
            self.y_test = y_test

            self.train_sample_num.config(text=len(X_train))
            self.test_sample_num.config(text=len(x_test))

            if self.epoch_box.get() == '0':
                messagebox.showerror('showerror', 'Epoch must be greater than 0')
                return
            
            p.train(X_train, Y_train, self.lr, int(self.epoch_box.get()))
            print('Training Done')
            p.test(x_test, y_test)
            print('Testing Done')
            self.train_result = p.get_all_train_result()
            self.test_result = p.get_all_test_result()

            self.best_acc.config(text=str(p.get_best_test_acc()*100)+'%')
            self.best_epoch.config(text=str(p.get_best_epoch()))
            self.best_weight.config(text=str([float(round(w, 2)) for w in p.get_best_weight()]))

            self.show_result()
            self.acc_graph(p.get_all_train_result(), p.get_all_test_result())

            # Visualize Results
            self.training_animation()
            self.load_btn.config(state='normal')
            self.select_btn.config(state='normal')
            self.train_btn.config(state='normal')
            self.save_btn.config(state='normal')
            self.show_update_btn.config(state='normal')
            self.stop_update_btn.place_forget()


        except Exception as e:
            print(f'Error training perceptron: {e}')
            return None
        
    def training_animation(self):
        try:
            self.stop_animation_flag = False
            for i in range(self.epoch):
                if self.stop_animation_flag:
                    break
                self.current_epoch.delete(0, tk.END)
                self.current_epoch.insert(0, i+1)
                self.current_result.config(text='Currently Showing result of epoch ' + str(self.current_epoch.get()))
                self.train_acc.config(text=str(self.train_result[i][1]*100)+'%')
                self.test_acc.config(text=str(self.test_result[i][1]*100)+'%')
                self.weight.config(text=str([float(round(w, 2)) for w in self.train_result[i][0]]))
                self.visualize(self.inputs, self.targets, 'Complete Data', self.canvas, self.train_result[i][0])
                self.visualize(self.x_train_without_x0, self.y_train, 'Training Data', self.train_canvas, self.train_result[i][0])
                self.visualize(self.x_test_without_x0, self.y_test, 'Testing Data', self.test_canvas, self.train_result[i][0])
        except Exception as e:
            self.stop_animation_flag = True
            print(f'Error showing the animation: {e}')
            return None

            
        
    def lrn_validation(self):
        self.lr = float(self.lrn_rate_box.get())
        if self.lr > 1 or self.lr < 0:
            messagebox.showerror('showerror', 'Learning Rate must be between 0 and 1')
            self.lr = 1.00
            self.lrn_rate_box.delete(0, tk.END)
            self.lrn_rate_box.insert(0, str(self.lr))
            return False
    def open(self):
        self.container.mainloop()

    def read_file(self, file):
        data = []
        self.train_result = []
        self.test_result = []
        self.clear_all_graph()
        self.train_sample_num.config(text='')
        self.test_sample_num.config(text='')
        try: 
            with open(file, "r") as f:
                all_lines = f.readlines() # readlines將每筆資料逐行讀取成list
                for line in all_lines:
                    line = line.strip('\n').split(' ') # strip把line break符號去掉, 然後用空格分割每筆資料
                    # Convert numpy array back to list
                    inputs = str2float(line[:-1]) # append每筆資料到input_data
                    targets = round(float(line[-1])) # append每筆資料到target_class
                    data.append([inputs, targets]) # append input 和 target as tuple
            self.data = data
            return data
        except Exception as e:
            self.data = None
            print(f'Error reading file: {e}')
            return None
        
    def acc_graph(self, train_result, test_result): 
        epoch = np.array(range(1, len(train_result)+1, 1))  # x軸: Epoch
        train_acc = np.array([i[1] * 100 for i in train_result])  # y軸: Training Accuracy as percentage
        test_acc = np.array([i[1] * 100 for i in test_result])  # y軸: Testing Accuracy as percentage

        fig = plt.Figure(figsize=(7, 1), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(epoch, train_acc, label='Train Accuracy')
        ax.plot(epoch, test_acc, label='Test Accuracy')
        
        ax.set_title('Training and Testing Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')  # Updated label to show percentage
        ax.legend()

        self.train_result_canvas = FigureCanvasTkAgg(master = self.result_frame)
        self.train_result_canvas.figure = fig
        self.train_result_canvas.get_tk_widget().config(width=500, height=310)
        self.train_result_canvas.get_tk_widget().place(x=340, y=5)

    def update_graph(self):
        epoch = int(self.current_epoch.get())
        if epoch > self.epoch:
            messagebox.showerror('showerror', 'Invalid Epoch')
            self.current_epoch.delete(0, tk.END)
            self.current_epoch.insert(0, self.epoch)
            return

        self.current_result.config(text='Currently Showing result of epoch ' + str(epoch))
        self.train_acc.config(text=str(self.train_result[epoch-1][1]*100)+'%')
        self.test_acc.config(text=str(self.test_result[epoch-1][1]*100)+'%')
        self.weight.config(text=str([float(round(w, 2)) for w in self.train_result[epoch-1][0]]))

        self.visualize(self.inputs, self.targets, 'Complete Data', self.canvas, self.train_result[epoch-1][0])
        self.visualize(self.x_train_without_x0, self.y_train, 'Training Data', self.train_canvas, self.train_result[epoch-1][0])
        self.visualize(self.x_test_without_x0, self.y_test, 'Testing Data', self.test_canvas, self.train_result[epoch-1][0])

    def show_result(self):
        self.current_result.place(x=0, y=0)
        self.current_epoch.place(x=0, y=30)
        self.show_update_btn.place(x=100, y=30)

        self.train_acc_label.place(x=0, y=80)
        self.train_acc.place(x=120, y=80)
        self.test_acc_label.place(x=0, y=100)
        self.test_acc.place(x=120, y=100)
        self.weight_label.place(x=0, y=120)
        self.weight.place(x=120, y=120)

        self.best_acc_label.place(x=0, y=160)
        self.best_acc.place(x=120, y=160)
        self.best_epoch_label.place(x=0, y=200)
        self.best_epoch.place(x=120, y=200)
        self.best_weight_label.place(x=0, y=220)
        self.best_weight.place(x=120, y=220)

    def hide_result(self):
        self.current_result.place_forget()
        self.current_epoch.place_forget()
        self.show_update_btn.place_forget()

        self.train_acc_label.place_forget()
        self.train_acc.place_forget()
        self.test_acc_label.place_forget()
        self.test_acc.place_forget()
        self.weight_label.place_forget()
        self.weight.place_forget()

        self.best_acc_label.place_forget()
        self.best_acc.place_forget()
        self.best_epoch_label.place_forget()
        self.best_epoch.place_forget()
        self.best_weight_label.place_forget()
        self.best_weight.place_forget()


    def clear_all_graph(self):
        # Reinitialize the canvas to make it a new graph
        self.canvas = FigureCanvasTkAgg(master = self.graph_frame)
        # self.canvas.figure = None
        self.canvas.draw()
        # Placing the canvas on graph_frame
        self.canvas.get_tk_widget().update()

        self.canvas.get_tk_widget().config(width=430, height=400)

        self.train_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.train_canvas.get_tk_widget().config(width=430, height=400)

        self.test_canvas = FigureCanvasTkAgg(master = self.graph_frame)
        self.test_canvas.get_tk_widget().config(width=430, height=400)
        
        self.train_result_canvas = FigureCanvasTkAgg(master = self.result_frame)
        self.train_result_canvas.get_tk_widget().config(width=500, height=310)
        
        self.canvas.get_tk_widget().place(x=0, y=10)
        self.train_canvas.get_tk_widget().place(x=450, y=10)
        self.test_canvas.get_tk_widget().place(x=890, y=10)
        self.train_result_canvas.get_tk_widget().place(x=340, y=5)
        self.hide_result()


    def init_all_member(self):
        self.file_name = ''
        self.data = None
        self.inputs = np.array([])
        self.targets = np.array([])

        self.epoch = 0
        self.lr = 0
        self.dim = 0
        self.epoch_result = 1
        self.best_epoch.config(text='')
        self.best_weight.config(text='')
        self.best_acc.config(text='')
        self.train_result = []
        self.test_result = []

        self.figure = None
        self.clear_all_graph()
        self.file_box.config(text='Current File: No File Selected')
        self.dim_text.config(text='')
        self.sample_num.config(text='')
        self.epoch_box.delete(0, tk.END)
        self.epoch_box.insert(0, '100')
        self.lrn_rate_box.delete(0, tk.END)
        self.lrn_rate_box.insert(0, '0.1')
        self.train_acc.config(text='...')
        self.test_acc.config(text='...')
        self.weight.config(text='...')
        self.train_sample_num.config(text='')
        self.test_sample_num.config(text='')

    def show_last_result(self):
        self.show_update_btn.config(state='normal')
        self.stop_update_btn.place_forget()
        self.current_epoch.delete(0, tk.END)
        self.current_epoch.insert(0, self.epoch)
        self.stop_animation_flag = True
        self.update_graph()
        self.show_result()
        self.show_update_btn.config(state='normal')
        self.load_btn.config(state='normal')
        self.select_btn.config(state='normal')
        self.train_btn.config(state='normal')
        self.save_btn.config(state='normal')
