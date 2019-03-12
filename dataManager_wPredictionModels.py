# GUI.
from PyQt5 import QtWidgets, QtGui, QtCore
import sys

# Regression Models.
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Data preprocessing.
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# File Op.
import datetime
import os

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        self.setWindowTitle("Modeling")
        self.theme()
        self.modelFeaturesDisabled()
        
    def init_ui(self):
        # Creating objects.
        self.data_label                = QtWidgets.QLabel("Data:")
        self.data_table                = QtWidgets.QTableWidget()
        self.inputColumn_y             = QtWidgets.QLineEdit()
        self.yColumn_label             = QtWidgets.QLabel("y Column:")
        self.createButton              = QtWidgets.QPushButton("Set Table")
        self.labelFileName             = QtWidgets.QLabel("File Name:")
        self.fileName                  = QtWidgets.QLineEdit()
        self.defineY_button            = QtWidgets.QPushButton("Define y Variable")
        self.defineX_button            = QtWidgets.QPushButton("Define x Variable")
        self.inputColumn_x             = QtWidgets.QLineEdit()
        self.xColumn_label             = QtWidgets.QLabel("x Column:")
        self.clearSelectionsButton     = QtWidgets.QPushButton("Clear Selections")
        self.infoLabel                 = QtWidgets.QLabel("Information line.")
        self.RegressionModels_combobox = QtWidgets.QComboBox()
        self.regressionLabel           = QtWidgets.QLabel("Regression Model:")
        self.createModel_button        = QtWidgets.QPushButton("Create Model")
        self.r2score_label             = QtWidgets.QLabel("R2 SCORE")
        self.r2score_value             = QtWidgets.QLabel("Selected Model:\n...\n\nScore:\n...")
        self.polyDegree_label          = QtWidgets.QLabel("Polynomial Regression Degree:")
        self.polyDegree_input          = QtWidgets.QLineEdit()
        self.svrKernel_label           = QtWidgets.QLabel("SVR Kernel:")
        self.svrKernel_combobox        = QtWidgets.QComboBox()
        self.rf_nestimators_label      = QtWidgets.QLabel("Random Forest n_estimators:")
        self.rf_nestimators_input      = QtWidgets.QLineEdit()
        self.originalData_label        = QtWidgets.QLabel("y_test (Original data):")
        self.originalData              = QtWidgets.QTableWidget()
        self.predictedData_label       = QtWidgets.QLabel("Predicted data:")
        self.predictedData             = QtWidgets.QTableWidget()
        self.emptylabel                = QtWidgets.QLabel("""Enter the .csv file name (exact name)\nClick 'Set Table' Define y variable and x variables.\nSelect a model, enter the features.\nClick 'Create Model'.""")
        self.emptylabel2               = QtWidgets.QLabel("You will see original data and predicted data at the right side.\nF1 = Print Corr.")
        self.printCorr_button          = QtWidgets.QPushButton("Print Corr")
        
        # Adding widgets and creating layouts.
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.data_label)
        vbox.addWidget(self.data_table)
        vbox.addWidget(self.labelFileName)
        vbox.addWidget(self.fileName)
        vbox.addWidget(self.createButton)
        vbox.addWidget(self.yColumn_label)
        vbox.addWidget(self.inputColumn_y)
        vbox.addWidget(self.defineY_button)
        vbox.addWidget(self.xColumn_label)
        vbox.addWidget(self.inputColumn_x)
        vbox.addWidget(self.defineX_button)
        vbox.addWidget(self.infoLabel)
        
        #V2
        vbox2 = QtWidgets.QVBoxLayout()
        vbox2.addWidget(self.regressionLabel)
        vbox2.addWidget(self.RegressionModels_combobox)
        vbox2.addWidget(self.svrKernel_label)
        vbox2.addWidget(self.svrKernel_combobox)
        vbox2.addWidget(self.polyDegree_label)
        vbox2.addWidget(self.polyDegree_input)
        vbox2.addWidget(self.rf_nestimators_label)
        vbox2.addWidget(self.rf_nestimators_input)
        vbox2.addWidget(self.createModel_button)
        vbox2.addWidget(self.printCorr_button)
        vbox2.addWidget(self.clearSelectionsButton)
        vbox2.addWidget(self.r2score_label)
        vbox2.addWidget(self.r2score_value)
        vbox2.addWidget(self.emptylabel)
        vbox2.addWidget(self.emptylabel2)
        
        #V3
        vbox3 = QtWidgets.QVBoxLayout()
        vbox3.addWidget(self.originalData_label)
        vbox3.addWidget(self.originalData)
        vbox3.addWidget(self.predictedData_label)
        vbox3.addWidget(self.predictedData)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)
        self.setLayout(hbox)
        self.show()
        

        self.RegressionModels_combobox.addItem("Select a Model")
        self.RegressionModels_combobox.addItems(["Linear Regression", "Polynomial Regression",
                                                 "Support Vector Regression", "Decision Tree Regression",
                                                 "Random Forest Regression"])
    
        self.svrKernel_combobox.addItem("Select a Kernel")
        self.svrKernel_combobox.addItems(["rbf", "linear", "poly"])
        
        # Buttons are connecting their functions.
        self.createButton.clicked.connect(self.creatingTable)
        self.defineY_button.clicked.connect(self.define_y)
        self.defineX_button.clicked.connect(self.define_x)
        self.clearSelectionsButton.clicked.connect(self.clear_selections)
        self.createModel_button.clicked.connect(self.define_model)
        self.RegressionModels_combobox.currentIndexChanged.connect(self.enableInputs)
        self.printCorr_button.clicked.connect(self.printCorr)
        
        # These lists will hold x and y variables (columns). Columns will be hold as DataFrame object.
        self.x_column_list = []
        self.y_column_list = []
        
    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Enter:
            self.creatingTable()
        elif e.key() == QtCore.Qt.Key_F1:
            self.printCorr()
        elif e.key() == QtCore.Qt.Key_Delete:
            self.clear_selections()
        
        
    def creatingTable(self):
        # Reading excel file. Filename is taking from user.
        #                                    \/ 
        self.data    = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)
        # It turns into a DataFrame.      /\  
        print(self.df_data.corr())
        # We define row and column values with shape method. 
        # With these values, we can create same table onto the ui. 
        self.df_row = self.df_data.shape[0] # row // [0] brought row value.
        self.df_col = self.df_data.shape[1] # column // [1] brought column value.
        
        self.data_table.setRowCount(self.df_row)
        self.data_table.setColumnCount(self.df_col)
        # Table has done. But there is no data. We adjusted only table shape according to data shape.
        
        # Simple for loops. Giving the range limit as df_row. (Last row of our data)
        # For example If the data has a shape that (28,5), that means there are 28 rows. So, our range is 28 in this for loop.
        # At 87. statement we're giving the column limit as range of for loop. In our example, It is 5.
        for rowIndex in range(0, self.df_row):
            for colIndex in range(0, self.df_col):
                cell = self.df_data.iat[rowIndex,colIndex] # Datum taking.
                self.data_table.setItem(rowIndex,colIndex, QtWidgets.QTableWidgetItem(str(cell)))
                #                               \/                                         \/
                #                       where datum will set                              datum   
                # With this one statement, our table turns into exact shape of excel file. Also,
                # all data will be put exact excel cell in this table.
                
        self.infoLabel.setText("Table has set.")
        # Giving information to user.
        
    def define_y(self):
        # File uploading.
        self.data    = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data) # File (excel) turns into a DataFrame.
        y_column     = self.inputColumn_y.text()  # holding selected y column index.
        
        for i in range(0, self.df_row):
            self.data_table.item(i, int(y_column)-1).setBackground(QtGui.QColor(12, 253, 0))
            #                    \/         \/
            #         i is holding row   Not will change
            #         index.             bc, we colored only 1 column.
            # We set a color for selected column.
        
        # Now we will use our lists. In this function, we're taking y column.
        self.selectedYcolumn = self.df_data.iloc[:,int(y_column)-1:int(y_column)]
        # Selected column has taken.
        
        # Added into the list.
        self.y_column_list.append(self.selectedYcolumn)
        
        #print(self.y_column_list)
        #print(type(self.y_column_list[0]))
        self.infoLabel.setText("Column " + y_column + " has selected and added as y variable.")
        # Giving information to user.
    def define_x(self):
        
        self.data    = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)
        x_column     = self.inputColumn_x.text()
        
        for i in range(0, self.df_row):
            self.data_table.item(i,int(x_column)-1).setBackground(QtGui.QColor(63, 173, 232))
        
        self.selectedXcolumn = self.df_data.iloc[:,int(x_column)-1:int(x_column)]
        self.x_column_list.append(self.selectedXcolumn)

        self.infoLabel.setText("Column " + x_column + " has selected and added as x variable.")
        
    def enableInputs(self):
        
        current_model_index = self.RegressionModels_combobox.currentIndex()
        # 2 -> Polynomial
        # 3 -> SVR
        # 5 -> Random Forest
        
        if current_model_index == 2:
            self.polyDegree_input.setEnabled(True)
            self.svrKernel_combobox.setEnabled(False)
            self.rf_nestimators_input.setEnabled(False)
            
        elif current_model_index == 3:
            self.svrKernel_combobox.setEnabled(True)
            self.polyDegree_input.setEnabled(False)
            self.rf_nestimators_input.setEnabled(False)
            
        elif current_model_index == 5:
            self.rf_nestimators_input.setEnabled(True)
            self.svrKernel_combobox.setEnabled(False)
            self.polyDegree_input.setEnabled(False)
        else:
            self.polyDegree_input.setEnabled(False)
            self.svrKernel_combobox.setEnabled(False)
            self.rf_nestimators_input.setEnabled(False)
            
    ###################### Model defining process beginning. ######################
            
    # Which model has selected, we have to define it.
    def define_model(self):
        
        selectedModel = self.RegressionModels_combobox.currentIndex()
        # 1 -> Linear
        # 2 -> Polynomial
        # 3 -> Support Vector R.
        # 4 -> Decision Tree R.
        # 5 -> Random Forest R.
        
        if selectedModel == 1:
            self.linear_reg_model()
        elif selectedModel == 2:
            self.polynomial_reg_model()
        elif selectedModel == 3:
            self.sv_reg_model()
        elif selectedModel == 4:
            self.decisionTree_reg_model()
        elif selectedModel == 5:
            self.randomForest_reg_model()
    
    # According to selected model, we're running the related function.
    
    def linear_reg_model(self):
        # Define the y and x variables. We take the data from lists which created before.
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        #Training / test split.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        # Model fitting.
        linear_reg = LinearRegression()
        linear_reg.fit(x_train,y_train)
        self.prediction = linear_reg.predict(x_test) # Prediction has created.
        
        #R^2 Score.
        self.r2Score_lin = r2_score(y_test, self.prediction)
        # Sending R^2 Score to r2_Score function. It will show the result.
        self.r2_Score(self.r2Score_lin)
        # To show the results, we're sending the original (y_test) and predicted data (self.prediction) to a function.
        self.showResults(y_test, self.prediction)
        self.infoLabel.setText("Linear Regression Model has created.")
        # Giving information to user.
        
    def polynomial_reg_model(self):
        
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        # We take the polynom degree from user,                  \/ right here.
        polynomial_reg = PolynomialFeatures(degree = int(self.polyDegree_input.text()) )
        x_poly = polynomial_reg.fit_transform(x)
        
        linear_reg = LinearRegression()
        linear_reg.fit(x_poly, y)
        self.prediction = linear_reg.predict(polynomial_reg.fit_transform(x_test))
        
        self.r2Score_poly = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_poly)
        
        self.showResults(y_test, self.prediction)
        self.infoLabel.setText("Polynomial Regression Model has created.")
        
    def sv_reg_model(self):
        
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        # For SVR Model, we've to scale the data.
        sc1 = StandardScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        scaled_y = sc2.fit_transform(y)
        
        # Giving scaled data for test/train process,          \/        \/  right here.
        x_train, x_test, y_train, y_test = train_test_split(scaled_x, scaled_y, test_size=0.33, random_state=0)
        
        # Define the kernel. We take from user that value.
        self.kernel = self.svrKernel_combobox.currentText()
        # Set the kernel value  \/           \/ with format method.
        sv_reg = SVR(kernel = '{}'.format(self.kernel))
        sv_reg.fit(scaled_x, scaled_y)
        self.prediction = sv_reg.predict(x_test)
        
        self.r2Score_sv = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_sv)
        
        self.showResults(y_test, self.prediction)
        self.infoLabel.setText("SV Regression Model has created.")


        
    def decisionTree_reg_model(self):
        
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        decisinTree_reg = DecisionTreeRegressor(random_state=0)
        decisinTree_reg.fit(x,y)
        self.prediction = decisinTree_reg.predict(x_test)
        
        self.r2Score_dt = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_dt)
        
        self.showResults(y_test, self.prediction)
        self.infoLabel.setText("Decision Tree R. Model has created.")
        
        
    def randomForest_reg_model(self):
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        # Taking the n_estimators value from user                              \/ right here.
        randomForest_reg = RandomForestRegressor(n_estimators = int(self.rf_nestimators_input.text()), random_state=0)
        randomForest_reg.fit(x, y)
        self.prediction = randomForest_reg.predict(x_test)

        self.r2Score_rf = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_rf)
        
        self.showResults(y_test, self.prediction)
        self.infoLabel.setText("Random Forest R. Model has created.")
        
    ###################### Model defining process has done. ######################
    
    # R^2 Score calculating.
    def r2_Score(self, score):
        # To show the model name and R^2 Score, taking the model name from current combobox text (current selected model)
        model_name = self.RegressionModels_combobox.currentText()
        # Showing it with a label and setText method.
        self.r2score_value.setText("Selected Model:\n{}\n\nScore:\n{}".format(model_name, str(score)))
    
    ###################### Results showing process is beginning. ######################
    
    def showResults(self, original_data, predicted_data):
        # Why do we have to use try-except? Because SVR, Random Forest and Decision Tree Regression models are
        # Giving the predicted data as a different dimension and as a numpy array. 
        # We cannot call the column value like 409.statment. We will take
        # out of bound (IndexError) If we'll call it like that. 
        # So, that structure isn't working  for every model. We're catching the IndexError and showing the values
        # according to their predicted data's shape and type.
        try:
            length = len(original_data)
            
            #Original data shape is defining.
            orgData_row = original_data.shape[0]
            orgData_col = original_data.shape[1]
            
            # According to its shape, creating the table shape.
            self.originalData.setRowCount(orgData_row)
            self.originalData.setColumnCount(orgData_col)
            
            #Predicted data shape is defining.
            predData_row = predicted_data.shape[0]
            predData_col = predicted_data.shape[1]
            
            # According to its shape, creating the table shape.
            self.predictedData.setRowCount(predData_row)
            self.predictedData.setColumnCount(predData_col)
            
            for i in range(0,length):
                # Original data's value.
                cell = original_data.iat[i,0]
                # Predicted data's value.
                cell2 = predicted_data[i][0]
                # Set the values into the table which we created before as same as predicted and original data's shape.
                self.originalData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell)))
                self.predictedData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell2)))
                # Some coloring.
                self.originalData.item(i,0).setBackground(QtGui.QColor(255,127,80))
                self.predictedData.item(i,0).setBackground(QtGui.QColor(255,215,0))
        # If we take IndexError (If SVR, RF or DT model selected)
        except IndexError:
            self.originalData.setRowCount(0)
            self.originalData.setColumnCount(0)
            # Clear the table.
            
            # Take the current model index.
            current_model = self.RegressionModels_combobox.currentIndex()
            
            # If its SVR
            if current_model == 3:
                self.svr_showingResults(original_data, predicted_data)
            else:
                self.rf_dt_showingResults(original_data, predicted_data)
            
            # SVR is giving the predicted data as a different dimension. But RF and DT are same.
            # So, there are two function one of them is for SVR and the other one is for RF and DT.
    def svr_showingResults(self, original_data, predicted_data):
        length = len(original_data)
        
        orgData_row = original_data.shape[0]
        orgData_col = original_data.shape[1]
            
        self.originalData.setRowCount(orgData_row)
        self.originalData.setColumnCount(orgData_col)
        
        # As you can see we're not calling the row value of predicted data.
        predData_row = predicted_data.shape[0]
        
        self.predictedData.setRowCount(predData_row)
        self.predictedData.setColumnCount(1)
        
        # Predicted and original data tables are setting.
        for i in range(0,length):
                cell = original_data[i][0] # In other models, original_data (self.originalData) comes as a DataFrame. Right here It's a numpy array.
                cell2 = predicted_data[i] # 1 dimension. (Look at 420.statement to see the difference.)
                self.originalData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell)))
                self.predictedData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell2)))
                self.originalData.item(i,0).setBackground(QtGui.QColor(255,127,80))
                self.predictedData.item(i,0).setBackground(QtGui.QColor(255,215,0))
                
    def rf_dt_showingResults(self, original_data, predicted_data):
        length = len(original_data)
        
        orgData_row = original_data.shape[0]
        orgData_col = original_data.shape[1]
            
        self.originalData.setRowCount(orgData_row)
        self.originalData.setColumnCount(orgData_col)
        
        # RF and DT models are giving different dimension value but also, (go 484.statement)
        predData_row = predicted_data.shape[0]
        
        self.predictedData.setRowCount(predData_row)
        self.predictedData.setColumnCount(1)
        
        for i in range(0,length):
                cell = original_data.iat[i,0] # It gives the the original data as a DataFrame (Like Linear and Polynomial) but,
                cell2 = predicted_data[i] # In here, Its 1 dimensional. Look at the 420.statement to see the difference.
                self.originalData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell)))
                self.predictedData.setItem(i,0, QtWidgets.QTableWidgetItem(str(cell2)))
                self.originalData.item(i,0).setBackground(QtGui.QColor(255,127,80))
                self.predictedData.item(i,0).setBackground(QtGui.QColor(255,215,0))
    
    ###################### Results showing process has done. ######################
    
    def printCorr(self):
        pd.set_option('display.expand_frame_repr', False)
        self.data = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)

        now = datetime.datetime.now()
        today_date = str(now).replace(":","-")
        file = open("{} Corr {}.txt".format(str(self.fileName.text()),today_date),"w+")
        file.write(str(self.df_data.corr()))
        os.startfile("{} Corr {}.txt".format(str(self.fileName.text()),today_date))
        
        
    def clear_selections(self):
        for i in range(0, self.df_row):
            for k in range(0, self.df_col):
                self.data_table.item(i, k).setBackground(QtGui.QColor(255, 255, 255))
        
        self.x_column_list.clear()
        self.y_column_list.clear()
        
        self.infoLabel.setText("Selections have deleted.")
    
    def modelFeaturesDisabled(self):
        self.polyDegree_input.setEnabled(False)
        self.svrKernel_combobox.setEnabled(False)
        self.rf_nestimators_input.setEnabled(False)
        
    def theme(self):
        # Font adjustments for table and other labels.
        self.tableFont = QtGui.QFont("Trebuchet MS", 10, QtGui.QFont.Bold)
        self.infoFont = QtGui.QFont("Trebuchet MS", 9, QtGui.QFont.Bold)
        self.TitleFont = QtGui.QFont("Trebuchet MS", 11, QtGui.QFont.Bold)
        self.r2score_titleFont = QtGui.QFont("Trebuchet MS", 12, QtGui.QFont.Bold)
        self.buttonsFont = QtGui.QFont("Trebuchet MS", 10, QtGui.QFont.Bold)
        
        self.infoLabel.setStyleSheet("QLabel {background : #464542; color: #FFD700;}")
        self.data_table.setFont(self.tableFont)
        self.infoLabel.setFont(self.infoFont)
        self.originalData.setFont(self.tableFont)
        self.predictedData.setFont(self.tableFont)
        
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.r2score_label.setAlignment(QtCore.Qt.AlignCenter)
        self.r2score_value.setAlignment(QtCore.Qt.AlignCenter)
        
        self.r2score_label.setFont(self.r2score_titleFont)
        self.r2score_value.setFont(self.TitleFont)
        
        self.labelFileName.setFont(self.TitleFont)
        self.yColumn_label.setFont(self.TitleFont)
        self.xColumn_label.setFont(self.TitleFont)
        self.regressionLabel.setFont(self.TitleFont)
        self.svrKernel_label.setFont(self.TitleFont)
        self.polyDegree_label.setFont(self.TitleFont)
        self.rf_nestimators_label.setFont(self.TitleFont)
        self.regressionLabel.setFont(self.TitleFont)
        self.originalData_label.setFont(self.TitleFont)
        self.predictedData_label.setFont(self.TitleFont)
        self.data_label.setFont(self.TitleFont)
        self.emptylabel.setFont(self.infoFont)
        self.emptylabel2.setFont(self.infoFont)
        
        self.defineX_button.setFont(self.buttonsFont)
        self.defineY_button.setFont(self.buttonsFont)
        self.createButton.setFont(self.buttonsFont)
        self.clearSelectionsButton.setFont(self.buttonsFont)
        self.printCorr_button.setFont(self.buttonsFont)
        self.createModel_button.setFont(self.buttonsFont)

        self.defineX_button.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
        self.defineY_button.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
        self.createButton.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
        self.clearSelectionsButton.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
        self.printCorr_button.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
        self.createModel_button.setStyleSheet("QPushButton {background: #008B8B; color: 'white';}")
            
        self.inputColumn_y.setStyleSheet("QLineEdit {background: #FFF8DC; color: 'black';}")
        self.fileName.setStyleSheet("QLineEdit {background: #FFF8DC; color: 'black';}")
        self.inputColumn_x.setStyleSheet("QLineEdit {background: #FFF8DC; color: 'black';}")
            
app = QtWidgets.QApplication(sys.argv)
window = Window()
window.move(200, 120)
window.setFixedSize(1000, 700)
app.setStyle("Fusion")
window.setStyleSheet("Window {background : #F0FFFF;}")
sys.exit(app.exec_())
