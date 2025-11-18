from PyQt5 import QtWidgets, QtCore, QtGui
from uidesigns.filters_tools_gui import Ui_FiltersTool
from uidesigns.resize_tools_gui import Ui_ResizeTool
from uidesigns.object_detection_tools_gui import Ui_ObjectDetectionTool


# from PyQt5.QtGui import QRegExpValidator
# from PyQt5.QtCore import QRegExp

# # Only positive integers (no leading zeros)
# regex = QRegExp("^[1-9][0-9]*$")
# validator = QRegExpValidator(regex)
# self.resize_widget.ui.width_resize_lineEdit.setValidator(validator)


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

class ResizeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ui = Ui_ResizeTool()
        self.ui.setupUi(self)

        int_validator = QtGui.QIntValidator()
        self.ui.width_resize_lineEdit.setValidator(int_validator)
        self.ui.height_resize_lineEdit.setValidator(int_validator)

        self.ui.content_aware_checkBox.stateChanged.connect(self.content_aware_check_state)
        self.content_aware_check_state()

        # Add aspect ratio options to combo box
        self.ui.resize_image_comboBox.clear()
        self.ui.resize_image_comboBox.addItems([
            "Original", 
            "1:1", 
            "4:3", 
            "3:4", 
            "16:9", 
            "9:16", 
            "3:2", 
            "2:3", 
            "Custom"
        ])

        # Dialog-like appearance and behavior
        self.setWindowFlags(
            QtCore.Qt.Tool | 
            QtCore.Qt.Dialog | 
            QtCore.Qt.WindowCloseButtonHint |
            QtCore.Qt.WindowTitleHint
        )
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setWindowModality(QtCore.Qt.NonModal)

    def content_aware_check_state(self):
        """
        Toggle text bold based on checkbox state using QFont
        """
        font = self.ui.content_aware_checkBox.font()
        font.setBold(self.ui.content_aware_checkBox.isChecked())
        self.ui.content_aware_checkBox.setFont(font)


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
