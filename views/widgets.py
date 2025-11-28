from typing import Union
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


from PyQt5 import QtWidgets, QtGui, QtCore
from typing import Union

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

        # Set up the QGraphicsView and QGraphicsScene for original image
        self.original_scene = QtWidgets.QGraphicsScene()
        self.ui.originalImagePreview.setScene(self.original_scene)
        
        # Set up the QGraphicsView and QGraphicsScene for processed image
        self.processed_scene = QtWidgets.QGraphicsScene()
        self.ui.processedImageView.setScene(self.processed_scene)

        self.all_graphics_views = [
            self.ui.originalImagePreview,
            self.ui.processedImageView
        ]
        
        # Set view properties for both views
        for view in self.all_graphics_views:
            view.setRenderHint(QtGui.QPainter.Antialiasing)
            view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(240, 240, 240)))
        
        # Initialize zoom combo box
        self.initialize_zoom_combo_box()
        
        # Connect the toggle scene button
        self.ui.toggleSceneBtn.clicked.connect(self.toggle_scene)
        
        # Add placeholder text
        self.show_placeholder_text()
        
        # Start with processed image view
        self.current_scene_index = 1  # 0 for processed, 1 for original
        self.ui.stackedWidget.setCurrentIndex(self.current_scene_index)
        self.update_toggle_button_text()

    @property
    def current_scene(self) -> QtWidgets.QGraphicsScene:
        """Get the current active scene based on current_scene_index"""
        if self.current_scene_index == 0:  # Processed image view
            return self.processed_scene
        else:  # Original image view
            return self.original_scene

    @property
    def current_graphics_view(self) -> QtWidgets.QGraphicsView:
        """Get the current active graphics view based on current_scene_index"""
        if self.current_scene_index == 0:  # Processed image view
            return self.ui.processedImageView
        else:  # Original image view
            return self.ui.originalImagePreview

    @property
    def current_dimensions_label(self) -> QtWidgets.QLabel:
        """Get the current dimensions label based on current_scene_index"""
        if self.current_scene_index == 0:  # Processed image view
            return self.ui.processedDimensionsLabel
        else:  # Original image view
            return self.ui.imageDimensionsLabel

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
        """Show placeholder text in both graphics views"""
        # Original image view
        self.original_scene.clear()
        original_text_item = self.original_scene.addText("No Original Image Selected")
        original_text_item.setDefaultTextColor(QtGui.QColor(100, 100, 100))
        font = original_text_item.font()
        font.setPointSize(14)
        original_text_item.setFont(font)
        
        # Processed image view
        self.processed_scene.clear()
        processed_text_item = self.processed_scene.addText("No Processed Image Available")
        processed_text_item.setDefaultTextColor(QtGui.QColor(100, 100, 100))
        processed_text_item.setFont(font)
        
        # Center the content
        self.center_content()

    def center_content(self):
        """Center the content in both graphics views"""
        self.ui.originalImagePreview.fitInView(self.original_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        self.ui.processedImageView.fitInView(self.processed_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def toggle_scene(self):
        """Toggle between original and processed image views"""
        # Toggle between 0 (processed) and 1 (original)
        self.current_scene_index = 1 - self.current_scene_index
        self.ui.stackedWidget.setCurrentIndex(self.current_scene_index)
        self.update_toggle_button_text()

    def update_toggle_button_text(self):
        """Update the toggle button text based on current scene"""
        if self.current_scene_index == 0:  # Currently showing processed image
            self.ui.toggleSceneBtn.setText("View Original")
        else:  # Currently showing original image
            self.ui.toggleSceneBtn.setText("View Processed")

    def show_original_image_page(self):
        """Switch to original image view when button is pressed"""
        self.ui.stackedWidget.setCurrentIndex(1)  # original_image_page
        self.current_scene_index = 1
        self.update_toggle_button_text()

    def show_processed_image_page(self):
        """Switch to processed image view when button is released"""
        self.ui.stackedWidget.setCurrentIndex(0)  # processed_image_page
        self.current_scene_index = 0
        self.update_toggle_button_text()

    def display_original_image(self, pixmap: Union[QtGui.QPixmap, None]):
        """Load and display original image"""
        if pixmap and not pixmap.isNull():
            self.original_scene.clear()
            original_pixmap_item = self.original_scene.addPixmap(pixmap)

            if original_pixmap_item:
                self.original_scene.setSceneRect(original_pixmap_item.boundingRect())
            self.ui.originalImagePreview.fitInView(self.original_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def display_processed_image(self, pixmap: Union[QtGui.QPixmap, None]):
        """Load and display processed image"""
        if pixmap and not pixmap.isNull():
            self.processed_scene.clear()
            processed_pixmap_item = self.processed_scene.addPixmap(pixmap)

            if processed_pixmap_item:
                self.processed_scene.setSceneRect(processed_pixmap_item.boundingRect())

            self.ui.processedImageView.fitInView(self.processed_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    # Helper methods using the current_scene property
    def clear_current_scene(self):
        """Clear the current active scene"""
        self.current_scene.clear()

    def add_item_to_current_scene(self, item):
        """Add an item to the current active scene"""
        return self.current_scene.addItem(item)

    def fit_current_view(self):
        """Fit the current graphics view to scene contents"""
        self.current_graphics_view.fitInView(self.current_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def update_current_dimensions(self, width: int, height: int):
        """Update the dimensions label for the current view"""
        self.current_dimensions_label.setText(f"{width} x {height}")
        


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
            "Hubble 003",
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
