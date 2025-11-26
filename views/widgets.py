from PyQt5 import QtWidgets, QtCore, QtGui
from uidesigns.main_window_gui import Ui_ImageLab
from uidesigns.filters_tools_gui import Ui_FiltersTool
from uidesigns.resize_tools_gui import Ui_ResizeTool
from uidesigns.object_detection_tools_gui import Ui_ObjectDetectionTool


# from PyQt5.QtGui import QRegExpValidator
# from PyQt5.QtCore import QRegExp

# # Only positive integers (no leading zeros)
# regex = QRegExp("^[1-9][0-9]*$")
# validator = QRegExpValidator(regex)
# self.resize_widget.ui.width_resize_lineEdit.setValidator(validator)


class ImageLabMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_ImageLab()
        self.ui.setupUi(self)

        self.initialize_ui()

    def initialize_ui(self):
        """Initialize UI settings for QGraphicsView and zoom controls"""

        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)


        # Set up the QGraphicsView and QGraphicsScene
        self.scene = QtWidgets.QGraphicsScene()
        self.ui.ImagePreview.setScene(self.scene)
        
        # Set view properties
        self.ui.ImagePreview.setRenderHint(QtGui.QPainter.Antialiasing)
        self.ui.ImagePreview.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        self.ui.ImagePreview.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        
        # Set alignment and background
        self.ui.ImagePreview.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.ImagePreview.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
        
        # Initialize zoom combo box
        self.initialize_zoom_combo_box()
        
        # Add placeholder text
        self.show_placeholder_text()
        
    
    def initialize_zoom_combo_box(self):
        """Initialize the zoom combo box with standard options"""
        zoom_options = [
            "Fit to View",
            "Actual Size", 
            "25%",
            "50%", 
            "75%",
            "100%",
            "125%",
            "150%",
            "200%",
            "300%",
            "400%",
            "Custom..."
        ]
        
        self.ui.selectStandardZoomComboBox.clear()
        self.ui.selectStandardZoomComboBox.addItems(zoom_options)
        self.ui.selectStandardZoomComboBox.setCurrentText("Fit to View")

    def show_placeholder_text(self):
        """Show placeholder text in the graphics view"""
        self.scene.clear()
        text_item = self.scene.addText("No Image Selected")
        text_item.setDefaultTextColor(QtGui.QColor(100, 100, 100))
        font = text_item.font()
        font.setPointSize(14)
        text_item.setFont(font)
        
        # Center the text
        self.center_content()

    def center_content(self):
        """Center the content in the graphics view"""
        self.ui.ImagePreview.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        


class ResizeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ui = Ui_ResizeTool()
        self.ui.setupUi(self)

        # Input validation
        int_validator = QtGui.QIntValidator(1, 100000)  # Add reasonable bounds
        self.ui.width_resize_lineEdit.setValidator(int_validator)
        self.ui.height_resize_lineEdit.setValidator(int_validator)

        # Connect signals
        self.ui.content_aware_checkBox.stateChanged.connect(self.content_aware_check_state)
        self.ui.content_aware_checkBox.toggled.connect(self.toggle_algorithm_visibility)
        
        # Initialize UI state
        self.content_aware_check_state()
        self.toggle_algorithm_visibility(self.ui.content_aware_checkBox.isChecked())

        # Aspect ratio options
        self.ui.resize_image_comboBox.clear()
        self.ui.resize_image_comboBox.addItems([
            "Custom",
            "Original", 
            "1:1", 
            "4:3", 
            "3:4", 
            "16:9", 
            "9:16", 
            "3:2", 
            "2:3",
        ])

        # Aspect ratio options
        self.ui.resize_algorithm_comboBox.clear()
        self.ui.resize_algorithm_comboBox.addItems([
            "Hubble 001",
            "Hubble 002",
        ])

        # Window configuration
        self.setWindowFlags(
            QtCore.Qt.Tool | 
            QtCore.Qt.Dialog | 
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowTitleHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowModality(QtCore.Qt.NonModal)

    def get_resize_parameters(self):
        """Get all resize parameters from the UI"""
        return {
            'width': int(self.ui.width_resize_lineEdit.text()) if self.ui.width_resize_lineEdit.text() else None,
            'height': int(self.ui.height_resize_lineEdit.text()) if self.ui.height_resize_lineEdit.text() else None,
            'content_aware': self.ui.content_aware_checkBox.isChecked(),
            'algorithm': self.ui.resize_algorithm_comboBox.currentText() if self.ui.content_aware_checkBox.isChecked() else None,
            'aspect_ratio': self.ui.resize_image_comboBox.currentText()
        }

    def reset_form(self):
        """Reset all form fields to default values"""
        self.ui.width_resize_lineEdit.clear()
        self.ui.height_resize_lineEdit.clear()
        self.ui.content_aware_checkBox.setChecked(True)
        self.ui.resize_algorithm_comboBox.setCurrentIndex(0)
        self.ui.resize_image_comboBox.setCurrentIndex(0)    

    def content_aware_check_state(self):
        """
        Toggle text bold based on checkbox state using QFont
        """
        font = self.ui.content_aware_checkBox.font()
        font.setBold(self.ui.content_aware_checkBox.isChecked())
        self.ui.content_aware_checkBox.setFont(font)

    # Add this method to your class:
    def toggle_algorithm_visibility(self, checked):
        """Show/hide algorithm selection based on content-aware checkbox state"""
        self.ui.algorithmWidget.setVisible(checked)

class FilterWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ui = Ui_FiltersTool()
        self.ui.setupUi(self)

        # Dialog-like appearance and behavior
        self.setWindowFlags(
            QtCore.Qt.Tool | 
            QtCore.Qt.Dialog | 
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowTitleHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowModality(QtCore.Qt.NonModal)


class ObjectDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ui = Ui_ObjectDetectionTool()
        self.ui.setupUi(self)

        self.ui.boundinBoxPreviewCheckbox.stateChanged.connect(self.bound_box_preview_state)
        self.bound_box_preview_state()

        # Dialog-like appearance and behavior
        self.setWindowFlags(
            QtCore.Qt.Tool | 
            QtCore.Qt.Dialog | 
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowTitleHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowModality(QtCore.Qt.NonModal)

    def bound_box_preview_state(self):
        """
        Toggle text bold based on checkbox state using QFont
        """
        font = self.ui.boundinBoxPreviewCheckbox.font()
        font.setBold(self.ui.boundinBoxPreviewCheckbox.isChecked())
        self.ui.boundinBoxPreviewCheckbox.setFont(font)
