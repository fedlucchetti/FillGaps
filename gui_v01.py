import numpy as np
import matplotlib as plt
import math, sys, os
import nibabel as nib

 
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtWidgets import QFileDialog, QPushButton, QStyle, QSpinBox
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


from PyQt5.QtCore import Qt
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLVolumeItem
from pyqtgraph import InfiniteLine
from pyqtgraph import ImageView

from fillgaps.tools.utilities import Utilities

# from tools.ClickableIMG import ClickableImageView 

utils = Utilities()

# DATAPATH = "/Users/flucchetti/Documents/Connectonome/Data/MRSI_reconstructed/Basic"



def create_custom_colormap():
    # Create an array of colors (256 colors, RGBA format)
    colors = np.ones((256, 4), dtype=np.uint8) * 255
    colors[:, 0] = np.arange(256)  # Red channel (grayscale)
    colors[:, 1] = np.arange(256)  # Green channel (grayscale)
    colors[:, 2] = np.arange(256)  # Blue channel (grayscale)
    colors[0, :] = [255, 0, 0, 255]  # Set the first color (0) as red

    # Create and return a ColorMap object
    colormap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 256), color=colors)
    return colormap.getLookupTable()



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.DATAPATH = os.path.join(utils.DATAPATH,"MRSI_reconstructed")
        self.file_types = ["Qmask", "Conc", "Basic", "Holes"]
        self.selectedPixels = set()  # Keep track of selected pixels
        print("Missing arguments... using default")
        # data = nib.load(os.path.join(self.DATAPATH,"Basic","Basic0001.nii")).get_fdata()
        data, header = utils.load_nii(file_type="Basic",id=1)
        self.unique_ids = utils.list_unique_ids(self.file_types)  # Assuming this method exists and returns a list of IDs
        # print(self.unique_ids)
        # print("data shape",data.shape)
        # print("data min,max",np.min(data),np.max(data))
        data = np.flip(data, axis=2)
        self.tensor3D_current=data
        self.current_axis = 0  # 0 for x-axis, 1 for y-axis, 2 for z-axis
        self.custom_colormap = create_custom_colormap()
        self.initUI()

    def initUI(self):
        # Init positions for sldiers
        init_pos = np.array([self.tensor3D_current.shape[0]/2,self.tensor3D_current.shape[1]/2,self.tensor3D_current.shape[2]/2]).astype(int)

        ##################### LHS Figure #####################
        self.imageView1 = pg.ImageView(self)
        self.imageView1.getImageItem().mousePressEvent = self.imageClicked
        self.slider1 = QSlider(Qt.Horizontal, self)
        self.slider1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.slider1.setValue(init_pos[0])
        self.spinBox1 = QSpinBox(self)
        self.spinBox1.setRange(0, self.tensor3D_current.shape[0] - 1)
        self.spinBox1.setValue(init_pos[0])
        self.spinBox1.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox1.setFixedWidth(self.spinBox1.sizeHint().width() + 20)  # Adjust widtg
         # Add  lines to the first image
        self.vLine1 = InfiniteLine(angle=90, movable=False, pen='g')  
        self.imageView1.addItem(self.vLine1)
        self.hLine1 = InfiniteLine(angle=0, movable=False, pen='g') 
        self.imageView1.addItem(self.hLine1)



        ##################### Middle Figure #####################
        self.imageView2 = pg.ImageView(self)
        self.slider2 = QSlider(Qt.Horizontal, self)
        self.slider2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.slider2.setValue(init_pos[1])
        self.spinBox2 = QSpinBox(self)
        self.spinBox2.setRange(0, self.tensor3D_current.shape[1] - 1)
        self.spinBox2.setValue(init_pos[1])
        self.spinBox2.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox2.setFixedWidth(self.spinBox2.sizeHint().width() + 20)  # Adjust width
        # Add  lines to the second image
        self.hLine2 = InfiniteLine(angle=0, movable=False, pen='g')  # Horizontal line, green
        self.imageView2.addItem(self.hLine2)
        self.vLine2 = InfiniteLine(angle=90, movable=False, pen='g')  # Horizontal line, green
        self.imageView2.addItem(self.vLine2)



        ##################### RHS Figure #####################
        self.imageView3 = pg.ImageView(self)
        self.slider3 = QSlider(Qt.Horizontal, self)
        self.slider3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.slider3.setValue(init_pos[2])
        self.spinBox3 = QSpinBox(self)
        self.spinBox3.setRange(0, self.tensor3D_current.shape[2] - 1)
        self.spinBox3.setValue(init_pos[2])
        self.spinBox3.setStyleSheet("QSpinBox { font-size: 36pt; text-align: center; }")
        self.spinBox3.setFixedWidth(self.spinBox2.sizeHint().width() + 20)  # Adjust width
        # Add  lines to the third image
        self.hLine3= InfiniteLine(angle=0, movable=False, pen='g')  # Horizontal line, green
        self.imageView3.addItem(self.hLine3)
        self.vLine3 = InfiniteLine(angle=90, movable=False, pen='g')  # Horizontal line, green
        self.imageView3.addItem(self.vLine3)



        #################### TABLE ####################
        vLayoutTable = QVBoxLayout()
        self.idTable = QTableWidget(self)
        self.idTable.setColumnCount(1)
        self.idTable.setHorizontalHeaderLabels(["Patient IDs"])
        self.populate_id_table()
        self.idTable.cellClicked.connect(self.table_select)
        vLayoutTable.addWidget(self.idTable)
        

        ##################### Connectors #####################
        self.slider1.valueChanged[int].connect(lambda value: self.updateFromSlider1(value))
        self.spinBox1.valueChanged[int].connect(self.updateFromSpinBox1)
        self.slider2.valueChanged[int].connect(lambda value: self.updateFromSlider2(value))
        self.spinBox2.valueChanged[int].connect(self.updateFromSpinBox2)
        self.slider3.valueChanged[int].connect(lambda value: self.updateFromSlider3(value))
        self.spinBox3.valueChanged[int].connect(self.updateFromSpinBox3)

        # self.idTable.cellClicked.connect(self.table_select)



        # vLayout3D = QVBoxLayout()
        # glWidget = self.create_3d_plot(self.tensor3D_current)
        # vLayout3D.addWidget(glWidget)
        
        ##################### Layout ALL #####################
        vLayout1 = QVBoxLayout()
        vLayout1.addWidget(self.imageView1)
        vLayout1.addWidget(self.slider1)
        vLayout1.addWidget(self.spinBox1)
        vLayout2 = QVBoxLayout()
        vLayout2.addWidget(self.imageView2)
        vLayout2.addWidget(self.slider2)
        vLayout2.addWidget(self.spinBox2)
        vLayout3 = QVBoxLayout()
        vLayout3.addWidget(self.imageView3)
        vLayout3.addWidget(self.slider3)
        vLayout3.addWidget(self.spinBox3)
        hLayoutMain = QHBoxLayout()
        hLayoutMain.addLayout(vLayoutTable)
        hLayoutMain.addLayout(vLayout1)
        hLayoutMain.addLayout(vLayout2)
        hLayoutMain.addLayout(vLayout3)
        vLayoutMain = QVBoxLayout()
        vLayoutMain.addLayout(hLayoutMain)
        # vLayoutMain.addLayout(vLayout3D)
        # Set the main layout
        widget = QWidget()
        widget.setLayout(vLayoutMain)
        self.setCentralWidget(widget)



        self.plot3DWindow = Plot3DWindow(self.tensor3D_current)
        self.plot3DWindow.show()

         ##################### First Update #####################
        self.hLine1.setPos(self.slider3.value())
        self.vLine1.setPos(self.slider2.value())
        self.hLine2.setPos(self.slider3.value())
        self.vLine2.setPos(self.slider1.value())
        self.hLine3.setPos(self.slider2.value())
        self.vLine3.setPos(self.slider1.value())
        self.updateImage(self.imageView1, init_pos[0], axis=0)
        self.updateImage(self.imageView2, init_pos[1], axis=1)
        self.updateImage(self.imageView3, init_pos[2], axis=2)

    #    # Button to open file explorer
    #     self.loadButton = QPushButton(self)
    #     self.loadButton.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
    #     self.loadButton.clicked.connect(self.openFileNameDialog)
    #     self.loadButton.move(10, 40)  # Adjust position as needed (e.g., lower)
    #     self.loadButton.show()

    def populate_id_table(self):
        unique_ids_info = self.unique_ids
        self.idTable.setColumnCount(5)
        self.idTable.setHorizontalHeaderLabels(["ID", "Qmask", "Conc", "Basic", "Holes"])
        self.idTable.setRowCount(len(unique_ids_info))

        for row, (id, file_info) in enumerate(unique_ids_info.items()):
            self.idTable.setItem(row, 0, QTableWidgetItem(str(id)))
            for col, file_type in enumerate(["Qmask", "Conc", "Basic", "Holes"], start=1):
                self.idTable.setItem(row, col, QTableWidgetItem('Yes' if file_info[file_type] else 'No'))


    def table_select(self, row, column):
        selected_id = self.idTable.item(row, 0).text()
        column_name = self.idTable.horizontalHeaderItem(column).text()
        print(f"Selected ID: {selected_id}, Column: {column_name}")
        self.tensor3D_current, self.header = utils.load_nii(file_type=column_name,id=int(selected_id))

        
        self.updateImage(self.imageView1, self.slider1.value(), axis=0)
        self.updateImage(self.imageView2, self.slider2.value(), axis=1)
        self.updateImage(self.imageView3, self.slider3.value(), axis=2)
        # You can add additional logic here to do something with the selected ID and column


    def imageClicked(self, event):
        # Convert the mouse click position to image coordinates
        pos = self.imageView1.getImageItem().mapFromScene(event.pos())
        x, y = int(pos.x()), int(pos.y())

        # Handle the click event, for example, print coordinates
        print(f"Clicked coordinates: x={x}, y={y}")

        # Call the original mouse press event of the parent to maintain default interactive behaviors
        self.imageView1.getImageItem().parent().mousePressEvent(event)

    def updateLine23(self, value):
        # Update horizontal line position based on first slider
        self.vLine2.setPos(value)
        self.vLine3.setPos(value)

    def updateLine13(self, value):
        # Update vertical line position based on second slider
        self.vLine1.setPos(value)
        self.hLine3.setPos(value)

    def updateLine12(self, value):
        # Update vertical line position based on second slider
        self.hLine1.setPos(value)
        self.hLine2.setPos(value)

    def updateFromSpinBox1(self, value):
        self.spinBox1.setValue(value)
        self.slider1.setValue(value)
        self.updateLine23(value)
        self.updateImage(self.imageView1, value, axis=0)
        
    def updateFromSlider1(self, value):
        self.spinBox1.setValue(value)
        self.updateLine23(value)
        self.updateImage(self.imageView1, value, axis=0)

    def updateFromSpinBox2(self, value):
        self.spinBox2.setValue(value)
        self.slider2.setValue(value)
        self.updateLine13(value)
        self.updateImage(self.imageView2, value, axis=1)

    def updateFromSlider2(self, value):
        self.spinBox2.setValue(value)
        self.updateLine13(value)
        self.updateImage(self.imageView2, value, axis=1)


    def updateFromSpinBox3(self, value):
        self.spinBox3.setValue(value)
        self.slider3.setValue(value)
        self.updateLine12(value)
        self.updateImage(self.imageView3, value, axis=2)

    def updateFromSlider3(self, value):
        self.spinBox3.setValue(value)
        # self.slider3.setValue(value)
        self.updateLine12(value)
        self.updateImage(self.imageView3, value, axis=2)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select a NIfTI file", self.tensor3D_currentPATH,
                                                  "NIfTI Files (*.nii);;All Files (*)", options=options)
        if fileName:
            print("Selected file:", fileName)
            self.loadNiiFile(fileName)

    def loadNiiFile(self, file_path):
        # Load the NIfTI file
        self.tensor3D_current = np.flip(nib.load(file_path).get_fdata(), axis=2)
        # Update the slider range
        self.slider.setRange(0, self.tensor3D_current.shape[self.current_axis] - 1)
        # Update the image
        self.updateImage(self.imageView1, 0, axis=0)
        self.updateImage(self.imageView2, 0, axis=1)
        self.updateImage(self.imageView3, 0, axis=2)



    def updateImage(self, imageView, slice_index, axis):
        if axis == 0:
            slice_data = self.tensor3D_current[slice_index, :, :]
        elif axis == 1:
            slice_data = self.tensor3D_current[:, slice_index, :]
        elif axis == 2:
            slice_data = self.tensor3D_current[:, :, slice_index]

        # Create masks for different data conditions
        nan_mask = np.isnan(slice_data)
        minus_one_mask = (slice_data == -1)
        valid_data_mask = ~nan_mask & ~minus_one_mask

        # Prepare an empty array for normalized data
        normalized_data = np.zeros_like(slice_data)

        # Normalize valid data (ignoring NaNs and -1 values)
        if np.any(valid_data_mask):
            valid_min = np.nanmin(slice_data[valid_data_mask])
            valid_max = np.nanmax(slice_data[valid_data_mask])
            normalized_data[valid_data_mask] = 255 * (slice_data[valid_data_mask] - valid_min) / (valid_max - valid_min) + 1

        # Apply custom colormap
        colored_slice = self.custom_colormap[normalized_data.astype(np.int32)]

        # Setting voxels with value -1 to green (green in RGBA is [0, 255, 0, 255])
        colored_slice[minus_one_mask] = [0, 255, 0]

        # Update the image view
        imageView.setImage(colored_slice)


class ClickableImageView(pg.ImageView):
    def __init__(self, *args, **kwargs):
        super(ClickableImageView, self).__init__(*args, **kwargs)
        self.selectedPixels = set()
        self.originalImageData = None  # To store the original image data

    def setImage(self, img, *args, **kwargs):
        self.originalImageData = np.array(img)
        super(ClickableImageView, self).setImage(img, *args, **kwargs)

    def updateImageWithSelection(self):
        if self.originalImageData is not None:
            # Create a copy of the original image to modify
            img_data = np.array(self.originalImageData)

            # Modify pixels based on selection
            for x, y in self.selectedPixels:
                if 0 <= x < img_data.shape[1] and 0 <= y < img_data.shape[0]:
                    img_data[y, x] = [0, 0, 255]  # Set selected pixels to blue

            # Update the displayed image
            super(ClickableImageView, self).setImage(img_data)


class Plot3DWindow(QMainWindow):
    def __init__(self, tensor3D):
        super().__init__()
        self.tensor3D = tensor3D
        self.initUI(tensor3D)

    def create_3d_plot(self, tensor3D):
        def normalize_data(data):
            nan_mask = np.isnan(data)
            min_val = np.nanmin(data)
            max_val = np.nanmax(data)
            data = (data - min_val) / (max_val - min_val)
            data[nan_mask] = 0  # You can choose how to handle NaNs
            return data

        tensor3D = normalize_data(tensor3D)
        # Create a GLViewWidget object
        self.glWidget = GLViewWidget()
        # Convert the 3D tensor to a format suitable for GLVolumeItem
        # Adjust this according to your tensor's format
        volume = np.flip(tensor3D, axis=0).T
        volume = np.ascontiguousarray(volume, dtype=np.float32)
        # Create a volume item
        self.volumeItem = GLVolumeItem(volume)
        self.volumeItem.translate(-volume.shape[0]/2, -volume.shape[1]/2, -volume.shape[2]/2)
        # Add the volume item to the widget
        self.glWidget.addItem(self.volumeItem)
        return self.glWidget

    def initUI(self,tensor3D):
        # Create a central widget
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        # Create a layout
        layout = QVBoxLayout(centralWidget)

        # Create the 3D plot
        glWidget = self.create_3d_plot(tensor3D)
        layout.addWidget(glWidget)




def main():
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    print("Main")
    main()




