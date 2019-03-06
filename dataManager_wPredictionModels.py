from PyQt5 import QtWidgets, QtGui, QtCore
import sys
import pandas as  pd
class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        
        self.init_ui()
        self.setWindowTitle("Modeling")
        self.theme()
    def init_ui(self):
        # Creating objects.
        self.data_table = QtWidgets.QTableWidget()
        self.inputColumn_y = QtWidgets.QLineEdit()
        self.labelCell_y = QtWidgets.QLabel("y Column:")
        self.createButton = QtWidgets.QPushButton("Set Table")
        self.labelFileName = QtWidgets.QLabel("File Name:")
        self.fileName = QtWidgets.QLineEdit()
        self.defineY_button = QtWidgets.QPushButton("Define y Variable")
        self.defineX_button = QtWidgets.QPushButton("Define x Variable")
        self.inputColumn_x = QtWidgets.QLineEdit()
        self.labelCell_x = QtWidgets.QLabel("x Column:")
        self.clearSelectionsButton = QtWidgets.QPushButton("Clear Selections")
        self.infoLabel = QtWidgets.QLabel("Information line.")
        self.RegressionModels_combobox = QtWidgets.QComboBox()
        self.regressionLabel = QtWidgets.QLabel("Regression Model:")
        
        # Adding widgets and creating layouts.
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.data_table)
        vbox.addWidget(self.labelCell_y)
        vbox.addWidget(self.inputColumn_y)
        vbox.addWidget(self.defineY_button)
        vbox.addWidget(self.labelCell_x)
        vbox.addWidget(self.inputColumn_x)
        vbox.addWidget(self.defineX_button)
        vbox.addWidget(self.regressionLabel)
        vbox.addWidget(self.RegressionModels_combobox)
        vbox.addWidget(self.labelFileName)
        vbox.addWidget(self.fileName)
        vbox.addWidget(self.createButton)
        vbox.addWidget(self.clearSelectionsButton)
        vbox.addWidget(self.infoLabel)
        
        hbox = QtWidgets.QHBoxLayout()
        hbox.addLayout(vbox)
        self.setLayout(hbox)
        self.show()
        
        self.infoLabel.setAlignment(QtCore.Qt.AlignCenter)
        
        # Buttons are connecting their functions.
        self.createButton.clicked.connect(self.creatingTable)
        self.defineY_button.clicked.connect(self.define_y)
        self.defineX_button.clicked.connect(self.define_x)
        self.clearSelectionsButton.clicked.connect(self.clear_selections)
        
        # These lists will hold x and y variables (columns). Columns will be hold as DataFrame object.
        self.x_column_list = []
        self.y_column_list = []
        
        self.RegressionModels_combobox.addItem("Select a Model")
        self.RegressionModels_combobox.addItems(["Linear Regression", "Polynomial Regression",
                                                 "Support Vector Regression", "Decision Tree Regression",
                                                 "Random Forest Regression"])
        
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
        
    def tempDefName(self):
        pass
        # Machine Learning operations will be happen here.
        # I will update soon.
        
    def clear_selections(self):
        for i in range(0, self.df_row):
            for k in range(0, self.df_col):
                self.data_table.item(i,k).setBackground(QtGui.QColor(255, 255, 255))
        
        self.x_column_list.clear()
        self.y_column_list.clear()
        
        self.infoLabel.setText("Selections have deleted.")
        
    def theme(self):
        # Font adjustments for table and other labels.
        self.tableFont = QtGui.QFont("Trebuchet MS", 12, QtGui.QFont.Bold)
        self.infoFont = QtGui.QFont("Trebuchet MS", 9, QtGui.QFont.Bold)
        
        self.infoLabel.setStyleSheet("QLabel {background : #464542; color: #FFD700;}")
        self.data_table.setFont(self.tableFont)
        self.infoLabel.setFont(self.infoFont)

app = QtWidgets.QApplication(sys.argv)
window = Window()
window.move(200, 120)
app.setStyle("Fusion")
sys.exit(app.exec_())