from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import pandas as  pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        self.setWindowTitle("Modeling")
        self.theme()
        self.modelFeaturesDisabled()
    def init_ui(self):
        # Creating objects.
        self.data_table = QtWidgets.QTableWidget()
        self.inputColumn_y = QtWidgets.QLineEdit()
        self.yColumn_label = QtWidgets.QLabel("y Column:")
        self.createButton = QtWidgets.QPushButton("Set Table")
        self.labelFileName = QtWidgets.QLabel("File Name:")
        self.fileName = QtWidgets.QLineEdit()
        self.defineY_button = QtWidgets.QPushButton("Define y Variable")
        self.defineX_button = QtWidgets.QPushButton("Define x Variable")
        self.inputColumn_x = QtWidgets.QLineEdit()
        self.xColumn_label = QtWidgets.QLabel("x Column:")
        self.clearSelectionsButton = QtWidgets.QPushButton("Clear Selections")
        self.infoLabel = QtWidgets.QLabel("Information line.")
        self.RegressionModels_combobox = QtWidgets.QComboBox()
        self.regressionLabel = QtWidgets.QLabel("Regression Model:")
        self.createModel_button = QtWidgets.QPushButton("Create Model")
        self.r2score_label = QtWidgets.QLabel("R2 Score:")
        self.r2score_value = QtWidgets.QLabel("Selected Model\n...")
        self.polyDegree_label = QtWidgets.QLabel("Polynomial Regression Degree:")
        self.polyDegree_input = QtWidgets.QLineEdit()
        self.svrKernel_label = QtWidgets.QLabel("SVR Kernel:")
        self.svrKernel_combobox = QtWidgets.QComboBox()
        self.rf_nestimators_label = QtWidgets.QLabel("Random Forest n_estimators:")
        self.rf_nestimators_input = QtWidgets.QLineEdit()
        
        # Adding widgets and creating layouts.
        vbox = QtWidgets.QVBoxLayout()
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
        vbox.addWidget(self.regressionLabel)
        vbox.addWidget(self.RegressionModels_combobox)
        vbox.addWidget(self.svrKernel_label)
        vbox.addWidget(self.svrKernel_combobox)
        vbox.addWidget(self.polyDegree_label)
        vbox.addWidget(self.polyDegree_input)
        vbox.addWidget(self.rf_nestimators_label)
        vbox.addWidget(self.rf_nestimators_input)
        vbox.addWidget(self.createModel_button)
        vbox.addWidget(self.clearSelectionsButton)
        vbox.addWidget(self.r2score_label)
        vbox.addWidget(self.r2score_value)
        vbox.addWidget(self.infoLabel)
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox)
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
        
        # These lists will hold x and y variables (columns). Columns will be hold as DataFrame object.
        self.x_column_list = []
        self.y_column_list = []

    def creatingTable(self):
        # Reading excel file. Filename is taking from user.
        #                                    \/ 
        self.data = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)
        # It turns into a DataFrame.      /\  
        
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
        
    def define_y(self):
        self.data = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)
        y_column = self.inputColumn_y.text()  # holding selected y column index.
        
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
    
    def define_x(self):
        self.data = pd.read_csv(str(self.fileName.text()) + '.csv')
        self.df_data = pd.DataFrame(self.data)
        x_column = self.inputColumn_x.text()
        
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
    
    def linear_reg_model(self):
        
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        linear_reg = LinearRegression()
        linear_reg.fit(x_train,y_train)
        self.prediction = linear_reg.predict(x_test)
        print(y_test)
        print(self.prediction)
        
        self.r2Score_lin = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_lin)
        self.infoLabel.setText("Linear Regression Model has created.")
        
    def polynomial_reg_model(self):
        
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        polynomial_reg = PolynomialFeatures(degree = int(self.polyDegree_input.text()) )
        x_poly = polynomial_reg.fit_transform(x)
        
        linear_reg = LinearRegression()
        linear_reg.fit(x_poly, y)
        self.prediction = linear_reg.predict(polynomial_reg.fit_transform(x_test))
        print(y_test)
        print(self.prediction)
        
        self.r2Score_poly = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_poly)
        self.infoLabel.setText("Polynomial Regression Model has created.")
        
    def sv_reg_model(self):
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        sc1 = StandardScaler()
        scaled_x = sc1.fit_transform(x)
        sc2 = StandardScaler()
        scaled_y = sc2.fit_transform(y)
        
        self.kernel = self.svrKernel_combobox.currentText()
        sv_reg = SVR(kernel = '{}'.format(self.kernel))
        sv_reg.fit(scaled_x, scaled_y)
        
        self.prediction = sv_reg.predict(x_test)
        print(y_test)
        print(self.prediction)
        
        self.r2Score_sv = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_sv)
        self.infoLabel.setText("SV Regression Model has created.")
        
    def decisionTree_reg_model(self):
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        decisinTree_reg = DecisionTreeRegressor(random_state=0)
        decisinTree_reg.fit(x,y)
        self.prediction =  decisinTree_reg.predict(x_test)
        print(y_test)
        print(self.prediction)
        
        self.r2Score_dt = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_dt)
        self.infoLabel.setText("Decision Tree R. Model has created.")
        
    def randomForest_reg_model(self):
        for j in range(0, len(self.x_column_list)):
            x = pd.DataFrame(data = self.x_column_list[j])
        for k in range(0, len(self.y_column_list)):
            y = pd.DataFrame(data = self.y_column_list[k])
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
        
        randomForest_reg = RandomForestRegressor(n_estimators = int(self.rf_nestimators_input.text()), random_state=0)
        randomForest_reg.fit(x, y)
        self.prediction = randomForest_reg.predict(x_test)
        print(y_test)
        print(self.prediction)
        
        self.r2Score_rf = r2_score(y_test, self.prediction)
        self.r2_Score(self.r2Score_rf)
        self.infoLabel.setText("Random Forest R. Model has created.")
        
    def r2_Score(self, score):
        model_name = self.RegressionModels_combobox.currentText()
        self.r2score_value.setText(model_name + "\n" + str(score))
        
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
        self.tableFont = QtGui.QFont("Trebuchet MS", 12, QtGui.QFont.Bold)
        self.infoFont = QtGui.QFont("Trebuchet MS", 9, QtGui.QFont.Bold)
        self.aTitleFont = QtGui.QFont("Trebuchet MS", 10, QtGui.QFont.Bold)
        
        self.infoLabel.setStyleSheet("QLabel {background : #464542; color: #FFD700;}")
        self.data_table.setFont(self.tableFont)
        self.infoLabel.setFont(self.infoFont)
        
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        
        self.r2score_label.setFont(self.aTitleFont)
        self.r2score_value.setFont(self.infoFont)
        

app = QtWidgets.QApplication(sys.argv)
window = Window()
window.move(200, 120)
app.setStyle("Fusion")
sys.exit(app.exec_())
