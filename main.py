import sys
import time
import csv
import statistics
import os
from numpy.polynomial import Polynomial
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.widgets import SpanSelector
import fitting

# This code automatically detects the names of the headers used to name the data, if the csv files don't contain
# such headers then the corresponding values can't be plotted

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("File choosing")
        self.setGeometry(200, 200, 400, 400)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.chooseManualMode()
        self.settings_window = SettingsWindow()
        self.defaultSettings()

        #self.folder_path = r'C:\Users\braun\PycharmProjects\pythonProject\ENAS_dots_cleaned2'
        #self.make_button()
        #self.load_list(r'C:\Users\braun\PycharmProjects\pythonProject\List.csv')

    def chooseAutoMode(self):
        for child in self.children():
            if isinstance(child, QPushButton) or isinstance(child, QTableWidget):
                child.hide()

        self.btn_choose_file = QPushButton("Choose list")
        self.btn_choose_file.clicked.connect(self.chooseList)
        self.layout.addWidget(self.btn_choose_file, 0, 0, 1, 1)

        self.btn_choose_folder = QPushButton("Choose folder")
        self.btn_choose_folder.clicked.connect(self.choosePath)
        self.layout.addWidget(self.btn_choose_folder, 0, 1, 1, 1)

        self.btn_selection_mode = QPushButton("Switch selection mode")
        self.btn_selection_mode.clicked.connect(self.chooseManualMode)
        self.layout.addWidget(self.btn_selection_mode, 0, 2, 1, 1)

        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.plotSettingsWindow)
        self.layout.addWidget(self.btn_settings, 0, 3, 1, 1)

        self.items_checked_bool = False

    def defaultSettings(self):
        self.prefix = "ENAS"
        self.suffix = "CCM"
        self.cv_suffix = "2"
        self.imp_suffix = "1"
        self.load_suffix = "1"

    def chooseList(self):
        List, _ = QFileDialog.getOpenFileName(self, "Choose CSV files", "", "CSV files (*.csv)")
        if List:
            self.load_list(List)

    def choosePath(self):
        # Choose the folder path containing the experiments
        self.folder_path = QFileDialog.getExistingDirectory(self, "Choose a folder", "")
        self.make_button()

    def chooseFiles(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Choose CSV files", "", "CSV files (*.csv)")
        if files:
            print(files)
            self.filenames = files
        self.makePlotButton()

    def makePlotButton(self):
        self.btn_plot = QPushButton("Plot data")
        self.btn_plot.clicked.connect(self.loadManualCSVData)
        self.layout.addWidget(self.btn_plot, 0, 1, 1, 1)

    def retrieveInputField(self):
        self.prefix = self.settings_window.prefix
        self.suffix = self.settings_window.suffix

        self.cv_suffix = self.settings_window.cv_suffix
        self.imp_suffix = self.settings_window.imp_suffix
        self.load_suffix = self.settings_window.load_suffix

    def chooseManualMode(self):
        for child in self.children():
            if isinstance(child, QPushButton) or isinstance(child, QTableWidget) or isinstance(child, QLineEdit):
                child.hide()

        self.btn_selection_mode = QPushButton("Switch selection mode")
        self.btn_selection_mode.clicked.connect(self.chooseAutoMode)
        self.layout.addWidget(self.btn_selection_mode, 0, 2, 1, 1)

        self.btn_choose_files = QPushButton("Choose files")
        self.btn_choose_files.clicked.connect(self.chooseFiles)
        self.layout.addWidget(self.btn_choose_files, 0, 0, 1, 1)

    def plotSettingsWindow(self):
        self.settings_window.show()

    def loadManualCSVData(self):
        filenames = self.filenames
        datasets_imp = []
        datasets_cv = []
        datasets_load = []

        filenames_imp = dict()
        filenames_cv = dict()
        filenames_load = dict()

        for filename in filenames:
            if "IMP" in filename.upper():
                dataset = self.load_dataset(filename, "IMP")
                datasets_imp.append(dataset)
                filenames_imp[os.path.basename(filename)] = []


            if "CV" in filename.upper():
                dataset = self.load_dataset(filename, "CV")
                datasets_cv.append(dataset)
                #filenames_cv[os.path.basename(filename)] = []
                filenames_cv[filename] = []

            if "LOAD" in filename.upper():
                dataset = self.load_dataset(filename, "LOAD")
                datasets_load.append(dataset)
                filenames_load[os.path.basename(filename)] = []

        # Modify to get the full folder path with the folder name so the folder path can be used in getOCCurrent, then take care of only displaying the filename
        if datasets_imp:
            self.plot_data(datasets_imp, "IMP", filenames_imp, "manual")

        if datasets_cv:
            self.plot_data(datasets_cv, "CV", filenames_cv, "manual")

        if datasets_load:
            self.plot_data(datasets_load, "LOAD", filenames_load, "manual")

    def load_list(self, listname):
        # Load the properties of the list of experiments
        try:
            with open(listname, "r", newline="") as file:
                reader = csv.DictReader(file)
                nb_lines = sum(1 for _ in reader)
                file.seek(0)
                column_names = [
                    "Original SAMPLE Nr.", "RAW LOAD", "RAW IMP",
                    "RAW CV", "Ink", "Catalyst", "Ionomer", "Solvent", "I:C ratio",
                    "Membrane", "Pt loading - anode/cathode - mg/cm2", "Comment", "Filename",
                    "A loading", "K loading", "total load"]

                nb_columns = len(column_names)
                column_indexes = []
                for col in column_names:
                    if col in reader.fieldnames:
                        column_indexes.append(reader.fieldnames.index(col))

                # Create a matrix containing the same elements as the list of experiments
                matrix = []
                file.seek(0)
                for row in reader:
                    columns = [row[field] for field in reader.fieldnames if
                               reader.fieldnames.index(field) in column_indexes]
                    matrix.append(columns)

                # Create the table
                self.table = QTableWidget(nb_lines, nb_columns)
                self.layout.addWidget(self.table, 3, 0, 1, 4)
                self.table.setHorizontalHeaderLabels(column_names)
                self.table.setVerticalHeaderLabels([''] * len(matrix))

                self.btn_select_all = QPushButton("Select all")
                self.btn_select_all.clicked.connect(self.select_all)
                self.layout.addWidget(self.btn_select_all, 4, 0, 1, 4)

                for row_index, row_data in enumerate(matrix[1:], start=1):
                    for col_index, data in enumerate(row_data[0:], start=1):
                        if col_index == 1:
                            # Make the box tickable if it is the first column
                            item = QTableWidgetItem(str(data))
                            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                            item.setCheckState(Qt.CheckState.Unchecked)
                            self.table.setItem(row_index - 1, col_index - 1, item)
                        else:
                            item = QTableWidgetItem(str(data))
                            self.table.setItem(row_index - 1, col_index - 1, item)

        except FileNotFoundError:
            print(f"File '{listname}' unknown.")

        except Exception as e:
            print(f"An error occurred: {e}")

    def make_button(self):
        # Create the button to plot impedance, load an cyclic voltammetry
        self.btn_plot_impedance = QPushButton("Plot Impedance")
        self.btn_plot_impedance.clicked.connect(lambda: self.retrieveCheckboxValues("IMP", self.folder_path))

        self.btn_plot_load = QPushButton("Plot Polarization Curve")
        self.btn_plot_load.clicked.connect(lambda: self.retrieveCheckboxValues("LOAD", self.folder_path))

        self.btn_plot_cv = QPushButton("Plot Cyclic Voltammetry")
        self.btn_plot_cv.clicked.connect(lambda: self.retrieveCheckboxValues("CV", self.folder_path))

        self.layout.addWidget(self.btn_plot_impedance, 2, 0)
        self.layout.addWidget(self.btn_plot_load, 2, 1)
        self.layout.addWidget(self.btn_plot_cv, 2, 2)

    def retrieveCheckboxValues(self, type, folder_path):
        self.retrieveInputField()
        self.checked_experiments = dict()
        for row in range(0, self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                self.checked_experiments[str(row + 1).zfill(2)] = None
        if type == "IMP":
            if hasattr(self, 'current_choosing_window') and self.current_choosing_window.isVisible():
                self.current_choosing_window.close()

            self.current_choosing_window = PlotCurrentChoosingWindow(self.checked_experiments, self.folder_path, self.suffix, self.prefix, self.imp_suffix)
            self.current_choosing_window.show()

        else:

            total_dict = self.create_experiment_dict(type)
            real_dict = dict()
            for key in self.checked_experiments.keys():
                if key in total_dict:
                    real_dict[key] = None

            self.plot_data(self.load_csv_data(real_dict, type), type, real_dict, "auto")

    def select_all(self):
        self.items_checked_bool = not self.items_checked_bool

        state = Qt.CheckState.Checked if self.items_checked_bool else Qt.CheckState.Unchecked
        for row in range(0, self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)
    def load_csv_data(self, checked_experiments: dict, type: str) -> list[list]:
        # Returns a list of lists containing all the datasets for a given experiment_list
        datasets = []
        # Browse the experiment's list

        if type == "IMP":
            for num, value in checked_experiments.items():
                # print("i current", i, current_list)
                # Get the filename of the experiment i
                if value != None:
                    for str in value:
                        filename = self.get_filename(num, type, str)
                        if not filename:
                            continue
                        dataset = self.load_dataset(filename, type)
                        if dataset != None:
                            datasets.append(dataset)

        else:
            print(checked_experiments.keys())
            for num in checked_experiments.keys():
                filename = self.get_filename(num, type, None)
                #print(filename)
                if not filename:
                    continue
                dataset = self.load_dataset(filename, type)
                if dataset != None:
                    datasets.append(dataset)

        return datasets

    def create_experiment_dict(self, type) -> dict:
        # Returns a dictionary which contains the currents available for each experiment

        experiment_dict = {}


        for filename in os.listdir(self.folder_path):

            if type == "CV":
                if type in filename and filename.endswith(f"_{self.cv_suffix}.csv"):
                    # Divide the filename

                    parts = filename.split('_')
                    if len(parts) >= 3:
                        experiment_number = ''.join(filter(str.isdigit, parts[2].split('-')[1]))

                        if experiment_number not in experiment_dict:
                            experiment_dict[experiment_number] = None
            if type == "LOAD":
                if type in filename and filename.endswith(f"_{self.load_suffix}.csv"):
                    # Divide the filename
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        experiment_number = ''.join(filter(str.isdigit, parts[2].split('-')[1]))

                        if experiment_number not in experiment_dict:
                            experiment_dict[experiment_number] = None

        return experiment_dict

    def get_filename(self, i: int, type: str, j: str) -> str:
        # Returns a filename for a given experiment number and a given current (used for PEIS)

        # Note : Find a solution to find automatically the prefix before -number_1 (if contains IMP and -number_1

        if type == "CV":
            return f"{self.folder_path}/{self.prefix}_CV_{self.suffix}-{str(i).zfill(2)}_{self.cv_suffix}.csv"

        elif type == "IMP":
            # Modify with current
            return f"{self.folder_path}/{self.prefix}_IMP_{self.suffix}-{str(i).zfill(2)}_{j}_{self.imp_suffix}.csv"

        elif type == "LOAD":
            return f"{self.folder_path}/{self.prefix}_LOAD_{self.suffix}-{str(i).zfill(2)}_{self.load_suffix}.csv"

        elif type == "CVShifting":
            return f"{self.folder_path}/{self.prefix}_CV_{self.suffix}-{str(i).zfill(2)}_b.csv"

        elif type == "CVShiftingmanual":
            return f"{self.folder_path}/{self.prefix}_CV_{self.suffix}-{str(i).zfill(2)}_b.csv"
        return None


    def load_dataset(self, filename: str, type: str) -> tuple[list[float]]:
        # Returns a list corresponding to the dataset of a given filename
        try:
            with (open(filename, "r", newline="") as file):
                reader = csv.reader(file, delimiter=',')

                # Skip metadata lines until we find the header line
                headers = next(reader)

                while not self.is_header(headers, type):
                    try:
                        headers = next(reader)
                    except StopIteration:
                        break

                column_indices = self.get_column_indices(headers, type)

                x_data = []
                y_data = []
                f_data = []
                if type == "IMP":
                    for row in reader:
                        if len(row) > max(column_indices):
                            f, x, y = float(row[column_indices[0]]), float(row[column_indices[1]]), float(row[column_indices[2]])
                            f_data.append(f)
                            x_data.append(x)
                            y_data.append(-y)

                    dataset = (f_data, x_data, y_data)
                    return dataset

                elif type == "CV" or type == "CVShifting":
                    for row in reader:
                        if len(row) > max(column_indices):
                            x, y = float(row[column_indices[0]]), float(row[column_indices[1]])
                            x_data.append(x)
                            y_data.append(y)

                    dataset = (x_data, y_data)
                    return dataset

                elif type == "LOAD":
                    for row in reader:
                        if len(row) > max(column_indices):
                            x, y = float(row[column_indices[0]]), float(row[column_indices[1]])
                            x_data.append(-x)
                            y_data.append(y)

                    dataset = (x_data, y_data)
                    return dataset

        except FileNotFoundError:
            print(f"File '{filename}' unknown.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def is_header(self, headers, type):
        if type == "IMP":
            return " Z' (Ohm)" in headers and " Z'' (Ohm)" in headers
        elif type == "LOAD":
            return " Current Density (A/cm2)" in headers and " Voltage (V)" in headers
        elif type == "CV":
            return " Voltage (V)" in headers and " Current Density (A/cm2)" in headers
        elif type == "CVShifting":
            return "Time (s)" in headers and " Current Density (A/cm2)" in headers
        return False

    def get_column_indices(self, headers, type):
        # Returns a couple of values corresponding to the column indices
        if type == "IMP":
            return headers.index(" Frequency (Hz)"), headers.index(" Z' (Ohm)"), headers.index(" Z'' (Ohm)")
        elif type == "LOAD":
            return headers.index(" Current Density (A/cm2)"), headers.index(" Voltage (V)")
        elif type == "CV":
            return headers.index(" Voltage (V)"), headers.index(" Current Density (A/cm2)")
        elif type == "CVShifting":
            return headers.index("Time (s)"), headers.index(" Current Density (A/cm2)")
        return None


    def plot_data(self, datasets, type, experiments, mode):
        if mode == "auto":
            if type == "IMP":
                self.window_imp = PlotWindow(datasets, "IMP", experiments, mode, self.folder_path, self.prefix, self.suffix)
                self.window_imp.show()

            elif type == "LOAD":
                self.window_load = PlotWindow(datasets, "LOAD", experiments, mode, self.folder_path, self.prefix, self.suffix)
                self.window_load.show()

            elif type == "CV":
                self.window_cv = PlotWindow(datasets, "CV", experiments, mode, self.folder_path, self.prefix, self.suffix)
                self.window_cv.show()
        if mode == "manual":
            if type == "IMP":
                self.window_imp = PlotWindow(datasets, "IMP", experiments, mode, None, None, None)
                self.window_imp.show()

            elif type == "LOAD":
                self.window_load = PlotWindow(datasets, "LOAD", experiments, mode, None, None, None)
                self.window_load.show()

            elif type == "CV":
                self.window_cv = PlotWindow(datasets, "CV", experiments, mode, None, None, None)
                self.window_cv.show()



class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Settings")
        self.setGeometry(200, 200, 400, 400)

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.prefix = "ENAS"
        self.prefix_field = QLineEdit(self)
        self.prefix_field.setText(self.prefix)
        self.layout.addWidget(self.prefix_field, 1, 0, 1, 1)

        self.suffix = "CCM"
        self.suffix_field = QLineEdit(self)
        self.suffix_field.setText(self.suffix)
        self.layout.addWidget(self.suffix_field, 1, 1, 1, 1)

        #Save settings
        self.save_settings_button = QPushButton("Save settings")
        self.save_settings_button.clicked.connect(self.saveSettings)
        self.layout.addWidget(self.save_settings_button, 1, 2, 1, 1)

        #Reset settings
        self.reset_settings_button = QPushButton("Reset settings")
        self.reset_settings_button.clicked.connect(self.resetSettings)
        self.layout.addWidget(self.reset_settings_button, 2, 2, 1, 1)

        #CV
        self.cv_label = QLabel("CV files end with : ", self)
        self.layout.addWidget(self.cv_label, 2, 0, 1, 1)

        self.cv_suffix = "2"
        self.cv_suffix_field = QLineEdit(self)
        self.cv_suffix_field.setText(self.cv_suffix)
        self.layout.addWidget(self.cv_suffix_field, 2, 1, 1, 1)

        #Impedance
        self.imp_label = QLabel("Impedance files end with : ", self)
        self.layout.addWidget(self.imp_label, 3, 0, 1, 1)

        self.imp_suffix = "1"
        self.imp_suffix_field = QLineEdit(self)
        self.imp_suffix_field.setText(self.imp_suffix)
        self.layout.addWidget(self.imp_suffix_field, 3, 1, 1, 1)

        #Load
        self.load_label = QLabel("Load files end with : ", self)
        self.layout.addWidget(self.load_label, 4, 0, 1, 1)

        self.load_suffix = "1"
        self.load_suffix_field = QLineEdit(self)
        self.load_suffix_field.setText(self.load_suffix)
        self.layout.addWidget(self.load_suffix_field, 4, 1, 1, 1)


    def saveSettings(self):
        self.prefix = self.prefix_field.text()
        self.suffix = self.suffix_field.text()
        self.cv_suffix = self.cv_suffix_field.text()
        self.imp_suffix = self.imp_suffix_field.text()
        self.load_suffix = self.load_suffix_field.text()

    def resetSettings(self):
        self.prefix = "ENAS"
        self.suffix = "CCM"
        self.cv_suffix = "2"
        self.imp_suffix = "1"
        self.load_suffix = "1"
        self.prefix_field.setText(self.prefix)
        self.suffix_field.setText(self.suffix)
        self.load_suffix_field.setText(self.load_suffix)
        self.imp_suffix_field.setText(self.imp_suffix)
        self.cv_suffix_field.setText(self.cv_suffix)

class PlotCurrentChoosingWindow(QWidget):
    def __init__(self, experiments, folder_path, suffix, prefix, imp_suffix):
        super().__init__()

        self.setWindowTitle("Current/Voltage choosing")

        self.main_window = MainWindow()
        self.main_window.suffix = suffix
        self.main_window.prefix = prefix

        self.main_window.folder_path = folder_path
        self.folder_path = folder_path
        self.imp_suffix = imp_suffix

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.setGeometry(400, 200, 200, 400)

        self.current_checked_bool = False

        total_dict = self.create_experiment_dict_imp()
        possible_currents = self.create_current_list(total_dict, experiments)

        nb_lines = len(possible_currents)
        nb_columns = 1

        self.table = QTableWidget(nb_lines, nb_columns)
        self.layout.addWidget(self.table, 4, 0, 1, 2)
        self.table.setHorizontalHeaderLabels(['Current'])
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setVerticalHeaderLabels([''] * len(possible_currents))


        for index, element in enumerate(possible_currents):
            item = QTableWidgetItem(str(element))
            item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.table.setItem(index, 0, item)

        self.btn_choose_current = QPushButton("Plot impedance")
        self.btn_choose_current.clicked.connect(lambda: self.retrieveCheckBoxCurrent(experiments.copy(), total_dict))

        self.btn_select_all_current = QPushButton("Select all")
        self.btn_select_all_current.clicked.connect(lambda: self.select_all_current())

        self.layout.addWidget(self.btn_choose_current, 0, 0, 1, 1)
        self.layout.addWidget(self.btn_select_all_current, 0, 1, 1, 1)


    def create_experiment_dict_imp(self) -> dict:
        # Returns a dictionary which contains the currents available for each experiment
        # Initialiser un dictionnaire vide pour stocker les résultats
        experiment_dict = {}

        # Lister tous les fichiers dans le dossier spécifié par 'path'
        for filename in os.listdir(self.folder_path):

            # Vérifier si le fichier suit le format attendu ENAS_IMP_CCM-XX_YYY_1
            if "IMP" in filename and filename.endswith(f"_{self.imp_suffix}.csv"):

                # Divide the filename
                parts = filename.split('_')
                if len(parts) >= 3:

                    # Extraire le numéro d'expérience (par exemple, 61 ou 62)
                    experiment_number = ''.join(filter(str.isdigit, parts[2].split('-')[1]))
                    #print("number:", experiment_number)
                    # Extraire le courant (par exemple, 125A, 375A, etc.)
                    current = parts[3]
                    current_with_suffix = parts[3]
                    #current_value = ''.join(filter(str.isdigit, current_with_suffix))
                    #suffix = ''.join(filter(lambda x: not x.isdigit(), current_with_suffix))

                    # Reformater le courant pour inclure le suffixe
                    #current = current_value + suffix

                    # Ajouter le courant dans la liste des valeurs pour ce numéro d'expérience
                    if experiment_number in experiment_dict:
                        experiment_dict[experiment_number].append(current)
                    else:
                        if current != None:
                            experiment_dict[experiment_number] = [current]
                        else:
                            experiment_dict[experiment_number] = ['None']
        return experiment_dict

    def create_current_list(self, dict: dict, experiments) -> list:
        # Returns a list which contains the currents available for the experiments selected
        possible_currents = []
        for exp, values in dict.items():
            if int(exp) in [int(exp) for exp in experiments]:
                for value in values:
                    if value not in possible_currents:
                        possible_currents.append(value)

        possible_currents = self.sort_by_suffix(possible_currents)
        return possible_currents

    def sort_by_suffix(self, lst):
        # Dictionnaire pour stocker les groupes selon les suffixes
        groups = {}

        # Séparer les éléments en groupes basés sur leurs suffixes
        for item in lst:
            # Extraire le suffixe en prenant la première partie non numérique
            suffix = ''.join(filter(lambda x: not x.isdigit(), item))
            number = ''.join(filter(str.isdigit, item))

            # Ajouter l'élément dans le groupe correspondant
            if suffix not in groups:
                groups[suffix] = []
            groups[suffix].append((int(number), item))

        # Trier chaque groupe par la partie numérique et recréer la liste triée
        sorted_list = []
        for suffix in sorted(groups):
            # Trier les éléments du groupe par la partie numérique
            groups[suffix].sort()
            # Ajouter les éléments triés au résultat final
            sorted_list.extend(item for _, item in groups[suffix])

        return sorted_list

    def retrieveCheckBoxCurrent(self, experiment_dict, total_dict):
        for row in range(0, self.table.rowCount()):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                for key in experiment_dict.keys():
                    if experiment_dict[key] == None:
                        # Adds the current to the dict if it isn't already in it
                        experiment_dict[key] = []
                    if key in total_dict and self.table.item(row, 0).text() in total_dict[key]:
                        experiment_dict[key].append(self.table.item(row, 0).text())

        self.main_window.plot_data(self.main_window.load_csv_data(experiment_dict, "IMP"), "IMP", experiment_dict, "auto")

    def select_all_current(self):
        self.current_checked_bool = not self.current_checked_bool

        state = Qt.CheckState.Checked if self.current_checked_bool else Qt.CheckState.Unchecked
        for row in range(0, self.table.rowCount()):
            self.table.item(row, 0).setCheckState(state)




class PlotWindow(QMainWindow, fitting.ImpedanceFitting, fitting.LOADFitting):
    def __init__(self, datasets, dataset_type, experiment_dict, mode, folder_path, prefix, suffix):
        super().__init__()
        super().__init__()

        self.main_window = MainWindow()
        self.main_window.suffix = suffix
        self.main_window.prefix = prefix

        self.main_window.folder_path = folder_path
        self.folder_path = folder_path

        self.datasets = datasets
        self.dataset_type = dataset_type
        self.experiment_dict = experiment_dict
        self.mode = mode
        self.prefix = prefix
        self.suffix = suffix

        self.fitting_datasets = []
        self.fitting_parameters = []

        self.setWindowTitle(dataset_type)
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.canvas = PlotCanvas()

        self.layout.addWidget(self.canvas, 2, 0, 1, 4)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar, 0, 0, 1, 4)

        self.set_axes_labels(dataset_type)
        self.plot_canvas_data(datasets, experiment_dict, dataset_type, mode, "line", "raw")
        self.make_button(datasets, experiment_dict, dataset_type)
        self.search_space_window = PlotSearchSpaceSettingsWindow()

    def changePlotType(self):
        if self.btn_radio5.isChecked():
            self.canvas.clearPlot()
            self.plot_canvas_data(self.datasets, self.experiment_dict, self.dataset_type, self.mode, "line", "raw")
        if self.btn_radio6.isChecked():
            self.canvas.clearPlot()
            self.plot_canvas_data(self.datasets, self.experiment_dict, self.dataset_type, self.mode, "scatter", "raw")

    def plot_canvas_data(self, datasets, experiment_dict, dataset_type, mode, plot_type, data_type):
        self.set_axes_labels(dataset_type)
        if dataset_type == "LOAD" or dataset_type == "CV" or dataset_type == "CVShifting" or dataset_type == "CVSlopeCorrected" or (mode == "manual" and data_type!="fit" and data_type != "IMP"):
            i = 0
            for exp_num in experiment_dict.keys():
                data_to_plot = datasets[i][0], datasets[i][1]
                self.x = datasets[i][0]
                self.y = datasets[i][1]
                label = None
                if mode == "auto":
                    label = f"Experiment {exp_num}"
                # Only for load and CV
                if mode == "manual":
                    label = os.path.basename(exp_num)
                if dataset_type == "IMP":
                    data_to_plot = datasets[i][1], datasets[i][2]
                    label = exp_num
                self.canvas.plot(*data_to_plot, plot_type, label)
                i += 1

        if mode == "manual" and data_type == "fit" and dataset_type == "IMP":
            i = 0
            for exp_num, fitness in experiment_dict.items():
                data_to_plot = datasets[i][0], datasets[i][1]
                self.x = datasets[i][0]
                self.y = datasets[i][1]
                label = None

                if dataset_type == "IMP":
                    data_to_plot = datasets[i][1], datasets[i][2]
                    label = f"{exp_num}, Fobj ={round(fitness,2)}"
                    #label = f"{exp_num}"
                self.canvas.plot(*data_to_plot, plot_type, label)
                i += 1

        if dataset_type == "IMP" and mode == "auto" and data_type == "raw":
            i = 0

            for exp_num, current_labels in experiment_dict.items():
                for current_label in current_labels:
                    data_to_plot = datasets[i][1], datasets[i][2]

                    label = f"Experiment {exp_num}, Current ={current_label}"

                    self.canvas.plot(*data_to_plot, plot_type, label)
                    i += 1
        if dataset_type == "IMP" and mode == "auto" and data_type == "fit":
            i = 0

            for exp_num, current_labels in experiment_dict.items():
                for current_label, fitness in current_labels.items():
                    data_to_plot = datasets[i][1], datasets[i][2]

                    label = f"Fitting {exp_num}, Current ={current_label}, Fobj = {round(fitness,2)}"

                    self.canvas.plot(*data_to_plot, plot_type, label)
                    i += 1

    def make_button(self, datasets, experiment_dict, dataset_type):
        if dataset_type == "IMP":
            button_title = "Fit Impedance Spectra"

            self.group1 = QButtonGroup(self)
            # Buttons to select the model to fit
            self.btn_radio1 = QRadioButton("R0 + R1/C1", self)
            self.btn_radio1.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            #self.layout.addWidget(self.btn_radio1)
            self.group1.addButton(self.btn_radio1)
            self.layout.addWidget(self.btn_radio1, 3, 0, 1, 1)

            self.btn_radio2 = QRadioButton("R0 + R1/Q1", self)
            self.btn_radio2.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            #self.layout.addWidget(self.btn_radio2)
            self.group1.addButton(self.btn_radio2)
            self.layout.addWidget(self.btn_radio2, 3, 1, 1, 1)

            self.btn_radio3 = QRadioButton("R0 + R1/C1 + R2/C2", self)
            self.btn_radio3.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            # self.layout.addWidget(self.btn_radio3)
            self.group1.addButton(self.btn_radio3)
            self.layout.addWidget(self.btn_radio3, 3, 2, 1, 1)

            self.btn_radio4 = QRadioButton("R0 + R1/Q1 + R2/Q2", self)
            self.btn_radio4.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            #self.layout.addWidget(self.btn_radio3)
            self.group1.addButton(self.btn_radio4)
            self.layout.addWidget(self.btn_radio4, 3, 3, 1, 1)
            # The last button is checked by default
            self.btn_radio4.setChecked(True)

            # Button to plot separate models
            self.btn_plot_separate_window = QPushButton("Plot Separate models")
            self.btn_plot_separate_window.clicked.connect(lambda: self.onButtonSeparateWindow(datasets, dataset_type))
            #self.btn_plot_separate_window.clicked.connect(lambda: self.onButtonFittingPlot(datasets, experiment_dict, "IMP"))
            self.layout.addWidget(self.btn_plot_separate_window, 1, 3, 1, 1)
            self.btn_plot_separate_window.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self.group2 = QButtonGroup(self)
            # Button to choose between line and dots
            self.btn_radio5 = QRadioButton("Line", self)
            self.btn_radio5.toggled.connect(lambda: self.changePlotType())
            self.group2.addButton(self.btn_radio5)
            self.layout.addWidget(self.btn_radio5, 4, 3, 1, 1)

            self.btn_radio6 = QRadioButton("Scatter", self)
            self.btn_radio6.toggled.connect(lambda: self.changePlotType())
            self.group2.addButton(self.btn_radio6)
            self.layout.addWidget(self.btn_radio6, 5, 3, 1, 1)


        if dataset_type == "LOAD":
            button_title = "Fit Polarisation Curve"
            self.btn_radio1 = QRadioButton("No limiting mass transport", self)
            self.btn_radio1.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            self.layout.addWidget(self.btn_radio1, 3, 0, 1, 1)

            self.btn_radio2 = QRadioButton("Limiting mass transport", self)
            self.btn_radio2.toggled.connect(lambda: self.onButtonRadioToggled(dataset_type))
            self.layout.addWidget(self.btn_radio2, 3, 1, 1, 1)

            self.btn_radio1.setChecked(True)



        if dataset_type == "IMP" or dataset_type == "LOAD":
            self.btn_fitting = QPushButton(button_title)
            self.btn_fitting.clicked.connect(lambda: self.onButtonFittingClick(datasets, experiment_dict, dataset_type))
            self.layout.addWidget(self.btn_fitting, 1, 0, 1, 1)
            self.btn_fitting.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self.btn_fitting_plot = QPushButton("Plot fitting datasets")
            self.btn_fitting_plot.clicked.connect(lambda: self.onButtonFittingPlot(datasets, experiment_dict, dataset_type))
            self.layout.addWidget(self.btn_fitting_plot, 1, 1, 1, 1)
            self.btn_fitting_plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self.fitness_label = QLabel("Stop on fitness value :", self)
            self.layout.addWidget(self.fitness_label, 4, 0, 1, 1)
            self.fitness_obj = 0.01
            self.fitness_obj_field = QLineEdit(self)
            self.fitness_obj_field.setText(str(self.fitness_obj))
            self.fitness_obj_field.textChanged.connect(self.update_fitness_obj)
            self.fitness_obj_field.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            self.layout.addWidget(self.fitness_obj_field, 4, 1, 1, 1)

            # Widget for choosing the optimization algorithm
            self.combo_box_label = QLabel("Optimization algorithm :", self)
            self.layout.addWidget(self.combo_box_label, 5, 0, 1, 1)
            self.combo_box = QComboBox()
            self.combo_box.addItem("BFGS algorithm")
            self.combo_box.addItem("Levenberg-Marquardt algorithm")
            self.combo_box.addItem("Genetic algorithm")
            self.combo_box.currentIndexChanged.connect(self.on_selection_change)
            self.optimization_algorithm = "bfgs"
            self.layout.addWidget(self.combo_box, 5, 1, 1, 1)

            # Widget for setting manually the search space
            self.btn_search_space = QPushButton("Search Space Settings")
            self.btn_search_space.clicked.connect(lambda: self.onButtonSearchSpaceClick(self.model))
            self.layout.addWidget(self.btn_search_space, 5, 2, 1, 1)
            self.btn_search_space.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


        if dataset_type == "CV":
            self.btn_shifting = QPushButton("Shift curve by OC value")
            self.btn_shifting.clicked.connect(lambda: self.onButtonShiftingClick(datasets, experiment_dict))
            self.layout.addWidget(self.btn_shifting, 1, 0, 1, 1)
            self.btn_shifting.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        if dataset_type == "CVShifting":
            self.btn_substract_slope = QPushButton("Correct curve by current linear increase")
            self.btn_substract_slope.clicked.connect(lambda: self.onButtonSubstractSlopeClick(datasets, experiment_dict, dataset_type))
            self.layout.addWidget(self.btn_substract_slope, 1, 0, 1, 1)
            self.btn_substract_slope.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

            self.btn_fitting = QPushButton("Calculate integral")
            self.btn_fitting.clicked.connect(lambda: self.onButtonFittingClick(datasets, experiment_dict, dataset_type))
            self.layout.addWidget(self.btn_fitting, 1, 1, 1, 1)

        if dataset_type != "CV":
            self.btn_saving = QPushButton("Save values")
            self.btn_saving.clicked.connect(lambda: self.onButtonSavingClick())
            self.layout.addWidget(self.btn_saving, 1, 2, 1, 1)
            self.btn_saving.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def retrieveSearchSpace(self):
        self.search_space_dict = self.search_space_window.search_space_dict

    def update_fitness_obj(self):
        try:
            self.fitness_obj = float(self.fitness_obj_field.text())
        except ValueError:
            # If the text cannot be converted to float, ignore it
            pass

    def on_selection_change(self):
        if self.combo_box.currentText() == "Genetic algorithm":
            self.optimization_algorithm = "GA"
        elif self.combo_box.currentText() == "BFGS algorithm":
            self.optimization_algorithm = "bfgs"
        elif self.combo_box.currentText() == "Levenberg-Marquardt algorithm":
            self.optimization_algorithm = "lm"

    def onButtonShiftingClick(self, datasets, experiment_dict):
        datasets_shifted = self.shiftDatasets(datasets, experiment_dict)
        #self.canvas.ax.clear()
        #self.plot_canvas_data(datasets_shifted, experiment_dict, "CV", "manual", "line", "fit")
        self.window_cv_shifted = PlotWindow(datasets_shifted, "CVShifting", self.experiment_dict, self.mode, self.folder_path, self.prefix, self.suffix)
        self.window_cv_shifted.show()

    def onButtonSearchSpaceClick(self, model):
        self.search_space_window.updateTable(model)
        self.search_space_window.show()

    def shiftDatasets(self, datasets, experiment_dict):
        mean = self.getOCCurrentValue(experiment_dict)
        datasets_shifted = []
        i = 0
        for dataset in datasets:
            dataset = (dataset[0], [x - mean[i] for x in dataset[1]])
            datasets_shifted.append(dataset)
            i += 1
        return datasets_shifted
    def getOCCurrentValue(self, experiment_dict):
        mean = []
        ocv_datasets = []
        for num in experiment_dict.keys():
            if self.mode == "auto":
                filename = self.main_window.get_filename(num, "CVShifting", None)
            elif self.mode == "manual":
                filename = num
            if filename:
                ocv_dataset = self.main_window.load_dataset(filename, "CVShifting")
            if not filename:
                continue
            if ocv_dataset != None:
                ocv_datasets.extend(ocv_dataset)
                mean.append(statistics.mean(ocv_dataset[1]))
        return mean

    def onButtonSubstractSlopeClick(self, datasets, experiment_dict, dataset_type):
        if hasattr(self.canvas,'xmin') and hasattr(self.canvas,'xmax'):
            datasets_shifted = datasets
            datasets_fitted = []
            datasets_corrected = []
            experiment_dict_fitted = dict()
            experiment_dict_corrected = dict()
            for dataset, key in zip(datasets_shifted, experiment_dict.keys()):
                x = np.array(dataset[0])
                y = np.array(dataset[1])

                x_pos = x[y > 0]
                y_pos = y[y > 0]
                # Find indices corresponding to values selected using the span selector
                ixmin = np.searchsorted(x, self.canvas.xmin)
                ixmax = np.searchsorted(x, self.canvas.xmax)
                # Isolate datasets within the interval
                x_sliced = x[ixmin: ixmax]
                y_sliced = y[ixmin: ixmax]
                # Calculate the polynomial
                p = Polynomial.fit(x_sliced, y_sliced, deg=1, domain=[self.canvas.xmin, self.canvas.xmax])
                # Append the coordinates to the dataset representing the polynom
                datasets_fitted.append((x_pos, (p(x_pos))))
                # Append the values corrected
                coeffs = p.convert().coef
                slope = coeffs[1]
                datasets_corrected.append((x, y - slope*x))
                experiment_dict_fitted[key + " Slope"] = []
                experiment_dict_corrected[key + " Corrected"] = []
            self.canvas.ax.clear()
            # Plot shifted curves
            self.plot_canvas_data(datasets_shifted, experiment_dict, dataset_type, self.mode, "line", "fit")
            # Plot regression line
            self.plot_canvas_data(datasets_fitted, experiment_dict_fitted, dataset_type, self.mode, "line", "fit")
            #Calculate integral
            self.plot_canvas_data(datasets_corrected, experiment_dict_corrected, dataset_type, self.mode, "line", "fit")
            #self.fitting_parameters = fitting.cv_integration(datasets_corrected, experiment_dict, self.mode, self.canvas.xmin, self.canvas.xmax)
        else:
            return
    def onButtonFittingClick(self, datasets, experiment_dict, dataset_type):
        if dataset_type == "IMP":

            self.retrieveSearchSpace()
            (num_generations,
             sol_per_pop,
             parent_selection_type,
             percent_parents_mating,
             crossover_type,
             crossover_probability,
             mutation_type,
             mutation_probability,
             keep_elitism,
             fitness_obj, IsInitialPop, fitness_coeff, fitness_function_type) = [100, 50, "tournament", 0.5, "scattered", 0.5, "adaptive", (0.5, 0.01), 2, 0.01, False, 1, "numpy"]

            (self.fitting_datasets,
             self.fitting_parameters,
             self.fitting_dict,
             self.time_list) = self.calculate_fit_imp(self.datasets,
                                                    self.search_space_dict,
                                                    experiment_dict,
                                                    self.mode,
                                                    self.model,
                                                    self.optimization_algorithm,
                                                    num_generations,
                                                    sol_per_pop,
                                                    parent_selection_type,
                                                    percent_parents_mating,
                                                    crossover_type,
                                                    crossover_probability,
                                                    mutation_type,
                                                    mutation_probability,
                                                    keep_elitism,
                                                    self.fitness_obj,
                                                    IsInitialPop,
                                                    fitness_coeff,
                                                    fitness_function_type)
        print(self.fitting_parameters)

        if dataset_type == "LOAD":

            self.retrieveSearchSpace()
            (num_generations,
             sol_per_pop,
             parent_selection_type,
             percent_parents_mating,
             crossover_type,
             crossover_probability,
             mutation_type,
             mutation_probability,
             keep_elitism,
             fitness_obj, IsInitialPop, fitness_coeff, fitness_function_type) = [100, 50, "tournament", 0.5,
                                                                                 "scattered", 0.5, "adaptive",
                                                                                 (0.5, 0.01), 2, 0.01, False, 1,
                                                                                 "numpy"]

            (self.fitting_datasets,
             self.fitting_parameters,
             self.fitting_dict,
             self.time_list) = self.calculate_fit_load(self.datasets,
                                                    self.search_space_dict,
                                                    experiment_dict,
                                                    self.mode,
                                                    self.model,
                                                    self.optimization_algorithm,
                                                    num_generations,
                                                    sol_per_pop,
                                                    parent_selection_type,
                                                    percent_parents_mating,
                                                    crossover_type,
                                                    crossover_probability,
                                                    mutation_type,
                                                    mutation_probability,
                                                    keep_elitism,
                                                    self.fitness_obj,
                                                    IsInitialPop,
                                                    fitness_coeff,
                                                    fitness_function_type)

        if dataset_type == "CVShifting":
            if self.canvas.xmin and self.canvas.xmax:
                self.fitting_parameters = fitting.cv_integration(datasets, experiment_dict, self.mode, self.canvas.xmin,
                                                                 self.canvas.xmax)
                # print(self.fitting_parameters)
            else:
                return

    def onButtonFittingPlot(self, datasets, experiment_dict, dataset_type):
        if dataset_type == "IMP":
            self.plot_canvas_data(self.fitting_datasets, self.fitting_dict, dataset_type, self.mode, "scatter", "fit")
        else:
            self.plot_canvas_data(self.fitting_datasets, experiment_dict, dataset_type, self.mode, "scatter", "fit")


    def onButtonSeparateWindow(self, datasets, dataset_type):
        if hasattr(self, 'model') and (self.model == "DoubleRQ" or self.model == "DoubleRC"):
            self.plotSeparatemodelFitting(datasets, self.fitting_dict, dataset_type, self.model)
            pass

    def plotSeparatemodelFitting(self, datasets, experiment_dict, dataset_type, model):
        mode = "manual"

        if mode == "manual":
            self.separate_windows = []
            for (i, dataset), (exp_num, fitness) in zip(enumerate(datasets), experiment_dict.items()):
                experiment_dict = {exp_num: fitness, 'Loop 1': [], 'Loop 2': []}

                separate_dataset = []
                separate_dataset.append(dataset)

                f = np.array(dataset[0])
                solution = self.fitting_parameters[i+1]

                if model == "DoubleRC":
                    R_el = solution[1]
                    R1 = solution[2]
                    C1 = solution[3]
                    R2 = solution[4]
                    C2 = solution[5]


                    Z1 = R1 / (1 + R1 * C1 * 1j * 2 * np.pi * f)
                    Z2 = R2 / (1 + R2 * C2 * 1j * 2 * np.pi * f)

                    if 1/(R1*C1)>1/(R2*C2):
                        Z1 += R_el
                        Z2 += R_el + R1
                    else:
                        Z1 += R_el + R2
                        Z2 += R_el

                if model == "DoubleRQ":
                    R_el = solution[1]
                    R1 = solution[2]
                    Q1 = solution[3]
                    alpha1 = solution[4]
                    R2 = solution[5]
                    Q2 = solution[6]
                    alpha2 = solution[7]

                    Z1 = R1 / (1 + R1 * Q1 * (1j * 2 * np.pi * f) ** alpha1)
                    Z2 = R2 / (1 + R2 * Q2 * (1j * 2 * np.pi * f) ** alpha2)

                    if 1/(R1*Q1)**(1/alpha1) > 1/(R2*Q2)**(1/alpha2):
                        Z1 += R_el
                        Z2 += R_el + R1
                    else:
                        Z1 += R_el + R2
                        Z2 += R_el


                R_num1 = Z1.real
                X_num1 = -Z1.imag

                R_num2 = Z2.real
                X_num2 = -Z2.imag

                separate_dataset.append((f, R_num1, X_num1))
                separate_dataset.append((f, R_num2, X_num2))

                self.separate_window = PlotSeparateWindow(separate_dataset, experiment_dict)
                self.separate_window.show()
                self.separate_windows.append(self.separate_window)


    def onButtonRadioToggled(self, dataset_type):
        if dataset_type == "IMP":
            if self.btn_radio1.isChecked():
                self.model = "Linear"
            if self.btn_radio2.isChecked():
                self.model = "RQ"
            if self.btn_radio3.isChecked():
                self.model = "DoubleRC"
            if self.btn_radio4.isChecked():
                self.model = "DoubleRQ"
            #print(self.model)

        if dataset_type == "LOAD":
            print("1 ok")
            if self.btn_radio1.isChecked():
                self.model = "NotLimited"
            if self.btn_radio2.isChecked():
                self.model = "Limited"

    def onButtonSavingClick(self):
        # Open file dialog for saving a file
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV Files (*.csv);;Text Files (*)")
        if filename and self.fitting_parameters:
            # Here you would save the file with the desired content
            # For example, writing a simple text to the file
            with open(filename, 'w', newline='') as file_out:
                writer = csv.writer(file_out)
                writer.writerows(self.fitting_parameters)

    def set_axes_labels(self, dataset_type):
        if dataset_type == "IMP":
            self.canvas.set_xlabel("Re(Z) / Ohm")
            self.canvas.set_ylabel("-Im(Z) / Ohm")
            self.canvas.orthonormal()
        elif dataset_type == "CV" or dataset_type == "CVShifting" or dataset_type == "CVSlopeCorrected":
            self.canvas.set_xlabel("Voltage / V")
            self.canvas.set_ylabel("Current density / A/cm^2")
        elif dataset_type == "LOAD":
            self.canvas.set_xlabel("Current density / A/cm^2")
            self.canvas.set_ylabel("Voltage / V")


class PlotSeparateWindow(QMainWindow, fitting.ImpedanceFitting, fitting.LOADFitting):
    def __init__(self, datasets, experiment_dict):
        super().__init__()

        #self.main_window = MainWindow()

        self.datasets = datasets

        self.experiment_dict = experiment_dict


        self.fitting_datasets = []
        self.fitting_parameters = []

        self.setWindowTitle("IMP")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QGridLayout()
        self.central_widget.setLayout(self.layout)

        self.canvas = PlotCanvas()

        self.layout.addWidget(self.canvas, 1, 0, 1, 5)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar, 0, 0, 1, 1)

        self.plot_canvas_data(datasets, experiment_dict, "line")


    def plot_canvas_data(self, datasets, experiment_dict, plot_type):
        self.set_axes_labels()

        i = 0
        for exp_num, fitness in experiment_dict.items():

            self.x = datasets[i][0]
            self.y = datasets[i][1]

            data_to_plot = datasets[i][1], datasets[i][2]

            label = f"{exp_num}"
            #print(data_to_plot)
            self.canvas.plot(*data_to_plot, plot_type, label)
            i += 1

    def set_axes_labels(self):

        self.canvas.set_xlabel("Re(Z) / Ohm")
        self.canvas.set_ylabel("-Im(Z) / Ohm")
        self.canvas.orthonormal()

class PlotSearchSpaceSettingsWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.search_space_dict = {'Linear': None, 'RQ': None, 'DoubleRC': None, 'DoubleRQ': None, 'NotLimited': None, 'Limited': None, 'FastSearch': True}

        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(lambda: self.onSaveButtonClick())
        self.layout.addWidget(self.save_button, 0, 0, 1, 1)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.save_button = QPushButton("Reset search space")
        self.save_button.clicked.connect(lambda: self.onResetButtonClick())
        self.layout.addWidget(self.save_button, 0, 1, 1, 1)
        self.save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.check_box = QCheckBox("Reduce resistance \n "
                                   "search space")
        self.check_box.stateChanged.connect(self.check_box_state_changed)
        self.check_box.setChecked(True)
        self.layout.addWidget(self.check_box, 0, 2, 1, 1)


    def updateTable(self, model):
        self.model = model
        if self.model == "Linear":
            rows = 3
            columns = 3
        elif self.model == "RQ":
            rows = 4
            columns = 3
        elif self.model == "DoubleRC":
            rows = 5
            columns = 3
        elif self.model == "DoubleRQ":
            rows = 7
            columns = 3
        elif self.model == "NotLimited":
            print("2 ok")
            rows = 5
            columns = 3
        elif self.model == "Limited":
            rows = 7
            columns = 3


        self.table = QTableWidget(rows, columns)
        self.table.setHorizontalHeaderLabels(["Parameters", "Lowest", "Highest"])
        self.table.setVerticalHeaderLabels(['']*rows)
        self.layout.addWidget(self.table, 1, 0, 1, 3)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setGeometry(400, 200, 300, int(rows/7*400))
        self.displayValues()


    def displayValues(self):
        if self.model == "Linear" or self.model == "RQ" or self.model == "DoubleRC" or self.model == "DoubleRQ":
            R_low = 0
            R_high = 0.1
            Q_low = 0.001
            Q_high = 1
            alpha_low = 0.5
            alpha_high = 1
        if self.model == "NotLimited" or self.model == "Limited":
            print("3 ok")
            In_low =  0.00001
            In_high = 0.1
            A_low = 0.000001
            A_high = 2
            I0_low = 0.0000001
            I0_high = 50
            E0_low = 0.1
            E0_high = 1.2
            R_low = 0.00001
            R_high = 10
            B_low = 0.0001
            B_high = 50
            Ilim_low = 2
            Ilim_high = 3000

        if self.model == "Linear":
            gene_names = ['R_el', 'R1', 'C1']
            if self.search_space_dict['Linear'] == None:
                search_space = [{'low': R_low, 'high': R_high},
                              {'low': R_low, 'high': R_high},
                              {'low': Q_low, 'high': Q_high}]
            else:
                search_space = self.search_space_dict['Linear']
        elif self.model == "RQ":
            gene_names = ['R_el', 'R1', 'Q1', 'alpha1']
            if self.search_space_dict['RQ'] == None:
                search_space = [{'low': R_low, 'high': R_high},
                          {'low': R_low, 'high': R_high},
                          {'low': Q_low, 'high': Q_high},
                          {'low': alpha_low, 'high': alpha_high}]
            else:
                search_space = self.search_space_dict['RQ']

        elif self.model == "DoubleRC":
            gene_names = ['R_el', 'R1', 'C1', 'R2', 'C2']
            if self.search_space_dict['RQ'] == None:
                search_space = [{'low': R_low, 'high': R_high},
                                {'low': R_low, 'high': R_high},
                                {'low': Q_low, 'high': Q_high},
                                {'low': R_low, 'high': R_high},
                                {'low': Q_low, 'high': Q_high}]
            else:
                search_space = self.search_space_dict['DoubleRC']

        elif self.model == "DoubleRQ":
            gene_names = ['R_el', 'R1', 'Q1', 'alpha1', 'R2', 'Q2', 'alpha2']
            print(self.search_space_dict)
            if self.search_space_dict['DoubleRQ'] == None:
                search_space = [{'low': R_low, 'high': R_high},
                              {'low': R_low, 'high': R_high},
                              {'low': Q_low, 'high': Q_high},
                              {'low': alpha_low, 'high': alpha_high},
                              {'low': R_low, 'high': R_high},
                              {'low': Q_low, 'high': Q_high},
                              {'low': alpha_low, 'high': alpha_high}]
            else:
                search_space = self.search_space_dict['DoubleRQ']

        elif self.model == "NotLimited":
            gene_names = ['In', 'A', 'I0', 'E0', 'R']
            print(self.search_space_dict)
            if self.search_space_dict['NotLimited'] == None:
                search_space = [{'low': In_low, 'high': In_high},
                              {'low': A_low, 'high': A_high},
                              {'low': I0_low, 'high': I0_high},
                              {'low': E0_low, 'high':E0_high},
                              {'low': R_low, 'high': R_high}]
            else:
                search_space = self.search_space_dict['NotLimited']

        elif self.model == "Limited":
            gene_names = ['In', 'A', 'I0', 'E0', 'R', 'B', 'Ilim']
            if self.search_space_dict['Limited'] == None:
                search_space = [{'low': In_low, 'high': In_high},
                                {'low': A_low, 'high': A_high},
                                {'low': I0_low, 'high': I0_high},
                                {'low': E0_low, 'high': E0_high},
                                {'low': R_low, 'high': R_high},
                                {'low': B_low, 'high': B_high},
                                {'low': Ilim_low, 'high': Ilim_high}]

            else:
                search_space = self.search_space_dict['Limited']


        for row_index in range(0, self.table.rowCount()):
            for col_index in range(0, self.table.columnCount()):
                if col_index == 0:
                    item = QTableWidgetItem(str(gene_names[row_index]))
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row_index, col_index, item)
                if col_index == 1:
                    item = QTableWidgetItem(str(search_space[row_index]['low']))
                    self.table.setItem(row_index, col_index, item)
                if col_index == 2:
                    item = QTableWidgetItem(str(search_space[row_index]['high']))
                    self.table.setItem(row_index, col_index, item)

    def onSaveButtonClick(self):
        search_space = []
        if self.model == 'Linear':
            self.makeZeroSearchSpace(search_space, 3)
        if self.model == 'RQ':
            self.makeZeroSearchSpace(search_space, 4)
        if self.model == 'DoubleRC':
            self.makeZeroSearchSpace(search_space, 5)
        if self.model == 'DoubleRQ':
            self.makeZeroSearchSpace(search_space, 7)
        if self.model == 'NotLimited':
            self.makeZeroSearchSpace(search_space, 5)
        if self.model == 'Limited':
            self.makeZeroSearchSpace(search_space, 7)

        for row_index in range(0, self.table.rowCount()):
            for col_index in range(0, self.table.columnCount()):
                search_space[row_index]['low'] = float(self.table.item(row_index, 1).text())
                search_space[row_index]['high'] = float(self.table.item(row_index, 2).text())

        if self.model == 'Linear':
            self.search_space_dict['Linear'] = search_space
        if self.model == 'RQ':
            self.search_space_dict['RQ'] = search_space
        if self.model == 'DoubleRC':
            self.search_space_dict['DoubleRC'] = search_space
        if self.model == 'DoubleRQ':
            self.search_space_dict['DoubleRQ'] = search_space
        if self.model == 'NotLimited':
            self.search_space_dict['NotLimited'] = search_space
        if self.model == 'Limited':
            self.search_space_dict['Limited'] = search_space

    def onResetButtonClick(self):
        if self.model == 'Linear':
            self.search_space_dict['Linear'] = None
        if self.model == 'RQ':
            self.search_space_dict['RQ'] = None
        if self.model == 'DoubleRC':
            self.search_space_dict['DoubleRC'] = None
        if self.model == 'DoubleRQ':
            self.search_space_dict['DoubleRQ'] = None
        if self.model == 'NotLimited':
            self.search_space_dict['NotLimited'] = None
        if self.model == 'Limited':
            self.search_space_dict['Limited'] = None

        self.displayValues()

    def makeZeroSearchSpace(self, search_space, num_param):
        for _ in range(num_param):
            search_space.append({'low': 0, 'high': 0})

    def check_box_state_changed(self, state):
        if state == 2:
            self.search_space_dict['FastSearch'] = True
        else:
            self.search_space_dict['FastSearch'] = False

class PlotCanvas(FigureCanvas):
    def __init__(self, parent=MainWindow, width=4, height=4, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)

    def plot(self, x_data, y_data, plot_type, label=None):
        if plot_type == "line":
            self.x = x_data
            self.y = y_data
            self.ax.plot(x_data, y_data, label=label, linewidth=8, zorder=1)
            self.span_selector = SpanSelector(self.ax, self.on_select, 'horizontal')
        elif plot_type == "scatter":
            self.ax.scatter(x_data, y_data, label=label, linewidth=4, zorder=2)

        if label:
            self.ax.legend()
        self.draw()

    def clearPlot(self):
        self.ax.clear()
    def on_select(self, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)

    def orthonormal(self):
        self.ax.set_aspect('equal', adjustable='box')
        #print(self.ax.get_xlim())
        #self.ax.set_xlim(left=0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
