"""
views/widgets.py
Widgets for ImageLab main window and tools
"""

from typing import Dict, Union, Optional, Tuple, List
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2
import numpy as np

from uidesigns.comparison_dialog import Ui_ComparisonDialog
from uidesigns.filters_tools_gui import Ui_FiltersTool
from uidesigns.main_window_gui import Ui_ImageLab
from uidesigns.object_detection_tools_gui import Ui_ObjectDetectionTool
from uidesigns.resize_tools_gui import Ui_ResizeTool


class ImageLabMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize image states BEFORE calling UI setup
        self.original_pixmap = None
        self.processed_pixmap = None
        self.seams_pixmap = None
        self.original_cv_image = None  # Store OpenCV format for processing

        self.ui = Ui_ImageLab()
        self.ui.setupUi(self)

        # Add missing UI elements from updated UI
        if hasattr(self.ui, 'compareAlgorithmsBtn'):
            self.ui.compareAlgorithmsBtn.setVisible(True)
        else:
            # Create if not present in UI
            self.ui.compareAlgorithmsBtn = QtWidgets.QPushButton("Compare Algorithms")
            self.ui.compareAlgorithmsBtn.setObjectName("compareAlgorithmsBtn")

        self.initialize_ui()

    def initialize_ui(self):
        """Initialize UI settings for QGraphicsView and zoom controls"""
        self.ui.progressBar.setVisible(False)
        self.ui.progressBar.setValue(0)
        
        # Initialize timing labels
        self.ui.timingLabel.setText("")
        self.ui.seamsTimingLabel.setText("")
        
        # Define aspect ratio presets
        self.aspect_ratios: Dict[str, Union[float, None]] = {
            "Original": None,
            "1:1": 1,
            "4:3": 4/3,
            "3:4": 3/4,
            "16:9": 16/9,
            "9:16": 9/16,
            "3:2": 3/2,
            "2:3": 2/3,
            "Custom": None
        }

        # Set up the QGraphicsView and QGraphicsScene for original image
        self.original_scene = QtWidgets.QGraphicsScene()
        self.ui.originalImagePreview.setScene(self.original_scene)
        
        # Set up the QGraphicsView and QGraphicsScene for processed image without seams
        self.processed_scene = QtWidgets.QGraphicsScene()
        self.ui.processedImageView.setScene(self.processed_scene)
        
        # Set up the QGraphicsView and QGraphicsScene for processed image with seams
        self.processed_with_seams_scene = QtWidgets.QGraphicsScene()
        self.ui.processedImageWithSeamsView.setScene(self.processed_with_seams_scene)

        self.all_graphics_views = [
            self.ui.originalImagePreview,
            self.ui.processedImageView,
            self.ui.processedImageWithSeamsView
        ]
        
        # Set view properties for all views
        for view in self.all_graphics_views:
            view.setRenderHint(QtGui.QPainter.Antialiasing)
            view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
            view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            view.setAlignment(QtCore.Qt.AlignCenter)
            view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(248, 250, 252)))
        
        # Initialize zoom combo box
        self.initialize_zoom_combo_box()
        
        # Initialize seams mode combo box
        self.initialize_seams_mode_combo_box()
        
        # Connect the toggle scene button
        self.ui.toggleSceneBtn.clicked.connect(self.toggle_scene)
        
        # Connect the seams view button
        self.ui.seamsViewBtn.clicked.connect(self.toggle_seams_view)
        
        # Connect seams mode combo box
        self.ui.seamsModeComboBox.currentTextChanged.connect(self.on_seams_mode_changed)
        
        # Add placeholder text
        self.show_placeholder_text()
        
        # Start with original image view (Page 0)
        self.current_page_index = 0  # 0=original, 1=processedWithoutSeams, 2=processedWithSeams
        self.ui.stackedWidget.setCurrentIndex(self.current_page_index)
        
        # Initialize seam viewing state
        self.seams_view_active = False
        self.current_seams_mode = "All Seams"  # Default to all seams
        self.seam_visualizations = None
        
        # Initialize timing info
        self.timing_info = None
        
        # Initialize button states - all disabled initially
        self.ui.toggleSceneBtn.setEnabled(False)
        self.ui.seamsViewBtn.setEnabled(False)
        self.ui.seamsModeComboBox.setVisible(False)
        
        # Update UI state
        self.update_toggle_button_text()

    @property
    def current_scene(self) -> QtWidgets.QGraphicsScene:
        """Get the current active scene based on current_page_index"""
        if self.current_page_index == 0:  # Original image
            return self.original_scene
        elif self.current_page_index == 1:  # Processed image without seams
            return self.processed_scene
        else:  # Processed image with seams (page index 2)
            return self.processed_with_seams_scene

    @property
    def current_graphics_view(self) -> QtWidgets.QGraphicsView:
        """Get the current active graphics view based on current_page_index"""
        if self.current_page_index == 0:  # Original image
            return self.ui.originalImagePreview
        elif self.current_page_index == 1:  # Processed image without seams
            return self.ui.processedImageView
        else:  # Processed image with seams (page index 2)
            return self.ui.processedImageWithSeamsView

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
            "400%"
        ]
        
        self.ui.selectStandardZoomComboBox.clear()
        self.ui.selectStandardZoomComboBox.addItems(zoom_options)
        self.ui.selectStandardZoomComboBox.setCurrentText("Fit to View")

    def initialize_seams_mode_combo_box(self):
        """Initialize the seams mode combo box"""
        self.ui.seamsModeComboBox.clear()
        self.ui.seamsModeComboBox.addItems(["All Seams", "Added Seams", "Removed Seams"])

    def show_placeholder_text(self):
        """Show placeholder text in all graphics views"""
        font = QtGui.QFont()
        font.setPointSize(12)
        
        # Original image view
        self.original_scene.clear()
        original_text_item = self.original_scene.addText("Upload or drag image here")
        original_text_item.setDefaultTextColor(QtGui.QColor(108, 117, 125))
        original_text_item.setFont(font)
        
        # Processed image view without seams
        self.processed_scene.clear()
        processed_text_item = self.processed_scene.addText("Processed image will appear here")
        processed_text_item.setDefaultTextColor(QtGui.QColor(108, 117, 125))
        processed_text_item.setFont(font)
        
        # Processed image view with seams
        self.processed_with_seams_scene.clear()
        seams_text_item = self.processed_with_seams_scene.addText("Seams visualization will appear here")
        seams_text_item.setDefaultTextColor(QtGui.QColor(108, 117, 125))
        seams_text_item.setFont(font)
        
        # Center the content
        self.center_content()

    def center_content(self):
        """Center the content in all graphics views"""
        for view, scene in [(self.ui.originalImagePreview, self.original_scene),
                           (self.ui.processedImageView, self.processed_scene),
                           (self.ui.processedImageWithSeamsView, self.processed_with_seams_scene)]:
            if scene.items():
                view.fitInView(scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)

    def toggle_scene(self):
        """Toggle between original and processed image views"""
        if not self.has_processed_image():
            return
            
        # If currently showing processed image with seams, switch to processed without seams first
        if self.current_page_index == 2:  # processed with seams
            self.ui.stackedWidget.setCurrentIndex(1)  # processed without seams (Page 1)
            self.current_page_index = 1
            self.seams_view_active = False
            self.ui.seamsViewBtn.setText("View Seams")
            self.ui.seamsModeComboBox.setVisible(False)
            
        # Toggle between original (0) and processed without seams (1)
        elif self.current_page_index == 0:  # Currently showing original
            self.ui.stackedWidget.setCurrentIndex(1)  # Switch to processed without seams
            self.current_page_index = 1
            # Show seam controls if available
            has_seams = self.seam_visualizations is not None
            self.ui.seamsViewBtn.setVisible(has_seams)
            self.ui.seamsModeComboBox.setVisible(False)  # Hide in processed without seams view
            
        else:  # Currently showing processed without seams (page index 1)
            self.ui.stackedWidget.setCurrentIndex(0)  # Switch to original
            self.current_page_index = 0
            # Hide seam controls when viewing original
            self.ui.seamsViewBtn.setVisible(False)
            self.ui.seamsModeComboBox.setVisible(False)
        
        # Update timing labels based on current page
        self.update_timing_display()
        self.update_toggle_button_text()

    def toggle_seams_view(self):
        """Toggle seam visualization on/off for processed image"""
        if not self.has_processed_image() or self.seam_visualizations is None:
            return
            
        self.seams_view_active = not self.seams_view_active
        
        if self.seams_view_active:
            # Turn ON seam view - switch to processed with seams page
            self.ui.stackedWidget.setCurrentIndex(2)  # processedImageWithSeamsPage (Page 2)
            self.current_page_index = 2
            self.ui.seamsViewBtn.setText("Hide Seams")
            self.ui.seamsModeComboBox.setVisible(True)
            self.show_processed_with_seams()
        else:
            # Turn OFF seam view - switch back to processed without seams page
            self.ui.stackedWidget.setCurrentIndex(1)  # processedImageWithoutSeamsPage (Page 1)
            self.current_page_index = 1
            self.ui.seamsViewBtn.setText("View Seams")
            self.ui.seamsModeComboBox.setVisible(False)
            self.show_processed_normal()
        
        # Update timing labels based on current page
        self.update_timing_display()
        self.update_toggle_button_text()

    def on_seams_mode_changed(self, mode_text):
        """Handle seam view mode change"""
        if not self.has_processed_image() or not self.seams_view_active or self.seam_visualizations is None:
            return
        
        self.current_seams_mode = mode_text
        self.show_processed_with_seams()

    def show_processed_normal(self):
        """Show processed image without seams (blended naturally)"""
        if self.processed_pixmap is not None:
            self.display_processed_image(self.processed_pixmap)

    def show_processed_with_seams(self):
        """Show processed image with seams based on current mode"""
        if self.seam_visualizations is None or not self.has_processed_image():
            return
        
        if self.current_seams_mode == "All Seams":
            self.show_all_seams()
        elif self.current_seams_mode == "Added Seams":
            self.show_added_seams()
        elif self.current_seams_mode == "Removed Seams":
            self.show_removed_seams()

    def show_all_seams(self):
        """Show processed image with ALL seams highlighted"""
        if self.seam_visualizations is not None and 'all' in self.seam_visualizations:
            seams_img = self.seam_visualizations['all']
            if seams_img is not None:
                q_image = self.convert_cv_to_qimage(seams_img)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.display_processed_with_seams_image(pixmap)

    def show_removed_seams(self):
        """Show processed image with removed seams highlighted"""
        if self.seam_visualizations is not None and 'removed' in self.seam_visualizations:
            seams_img = self.seam_visualizations['removed']
            if seams_img is not None:
                q_image = self.convert_cv_to_qimage(seams_img)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.display_processed_with_seams_image(pixmap)

    def show_added_seams(self):
        """Show processed image with added seams highlighted"""
        if self.seam_visualizations is not None and 'added' in self.seam_visualizations:
            seams_img = self.seam_visualizations['added']
            if seams_img is not None:
                q_image = self.convert_cv_to_qimage(seams_img)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.display_processed_with_seams_image(pixmap)

    def show_original_normal(self):
        """Show original image without seams"""
        if self.original_pixmap is not None:
            self.display_original_image(self.original_pixmap)

    def update_toggle_button_text(self):
        """Update the toggle button text based on current page"""
        if self.current_page_index == 0:  # Currently showing original
            self.ui.toggleSceneBtn.setText("View Processed")
        elif self.current_page_index == 1:  # Currently showing processed without seams
            self.ui.toggleSceneBtn.setText("View Original")
        else:  # Currently showing processed with seams (page 2)
            self.ui.toggleSceneBtn.setText("View Original")

    def update_button_states(self):
        """Update button states based on current state"""
        has_processed = self.has_processed_image()
        
        # Toggle button enabled only if we have processed image
        self.ui.toggleSceneBtn.setEnabled(has_processed)
        
        # Seams view button enabled only if we have processed image AND seam visualizations
        has_seams = self.seam_visualizations is not None
        self.ui.seamsViewBtn.setEnabled(has_processed and has_seams)
        
        # Show seams button only when in processed pages (1 or 2)
        in_processed_page = self.current_page_index in [1, 2]
        self.ui.seamsViewBtn.setVisible(in_processed_page and has_seams)
        
        # Show seams mode combo box only when viewing seams (page 2)
        self.ui.seamsModeComboBox.setVisible(self.current_page_index == 2)

    def show_original_image_page(self):
        """Switch to original image view"""
        self.ui.stackedWidget.setCurrentIndex(0)  # originalImagePage (Page 0)
        self.current_page_index = 0
        
        # Hide seam controls when viewing original
        self.ui.seamsViewBtn.setVisible(False)
        self.ui.seamsModeComboBox.setVisible(False)
        self.seams_view_active = False
        self.ui.seamsViewBtn.setText("View Seams")
        
        # Show original image normally
        self.show_original_normal()
        
        # Update button states
        self.update_toggle_button_text()
        self.update_button_states()
        
        # Clear timing display for original page
        self.ui.timingLabel.setText("")

    def show_processed_image_page(self):
        """Switch to processed image view without seams"""
        self.ui.stackedWidget.setCurrentIndex(1)  # processedImageWithoutSeamsPage (Page 1)
        self.current_page_index = 1
        
        # Show seam controls if available
        has_seams = self.seam_visualizations is not None
        self.ui.seamsViewBtn.setVisible(has_seams)
        self.ui.seamsModeComboBox.setVisible(False)  # Hidden in processed without seams view
        self.seams_view_active = False
        self.ui.seamsViewBtn.setText("View Seams")
        
        # Show processed image without seams
        self.show_processed_normal()
        
        # Update button states
        self.update_toggle_button_text()
        self.update_button_states()
        
        # Update timing display for processed page
        self.update_timing_display()

    def show_processed_with_seams_page(self):
        """Switch to processed image view with seams"""
        self.ui.stackedWidget.setCurrentIndex(2)  # processedImageWithSeamsPage (Page 2)
        self.current_page_index = 2
        
        # Show seam controls
        self.ui.seamsViewBtn.setVisible(True)
        self.ui.seamsModeComboBox.setVisible(True)
        self.seams_view_active = True
        self.ui.seamsViewBtn.setText("Hide Seams")
        
        # Show appropriate seam visualization
        self.show_processed_with_seams()
        
        # Update button states
        self.update_toggle_button_text()
        self.update_button_states()
        
        # Update timing display for seams page
        self.update_timing_display()

    def display_original_image(self, pixmap: Union[QtGui.QPixmap, None]):
        """Load and display original image"""
        if pixmap and not pixmap.isNull():
            self.original_scene.clear()
            original_pixmap_item = self.original_scene.addPixmap(pixmap)

            if original_pixmap_item:
                self.original_scene.setSceneRect(original_pixmap_item.boundingRect())
            self.ui.originalImagePreview.fitInView(self.original_scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
            
            # Store the pixmap
            self.original_pixmap = pixmap
            
            # Switch to original image page (Page 0)
            self.current_page_index = 0
            self.ui.stackedWidget.setCurrentIndex(self.current_page_index)
            
            # Update button states
            self.update_toggle_button_text()
            self.update_button_states()
            
            # Clear timing for original image
            self.ui.timingLabel.setText("")
            self.ui.seamsTimingLabel.setText("")

    def display_processed_image(self, pixmap: Union[QtGui.QPixmap, None]):
        """Load and display processed image without seams"""
        if pixmap and not pixmap.isNull():
            self.processed_scene.clear()
            processed_pixmap_item = self.processed_scene.addPixmap(pixmap)

            if processed_pixmap_item:
                self.processed_scene.setSceneRect(processed_pixmap_item.boundingRect())
            self.ui.processedImageView.fitInView(self.processed_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            # Store the pixmap
            self.processed_pixmap = pixmap
            
            # Switch to processed image page (Page 1)
            self.current_page_index = 1
            self.ui.stackedWidget.setCurrentIndex(self.current_page_index)
            
            # Update button states
            self.update_toggle_button_text()
            self.update_button_states()
            
            # Update timing display
            self.update_timing_display()

    def display_processed_with_seams_image(self, pixmap: Union[QtGui.QPixmap, None]):
        """Load and display processed image with seams visualization"""
        if pixmap and not pixmap.isNull():
            self.processed_with_seams_scene.clear()
            seams_pixmap_item = self.processed_with_seams_scene.addPixmap(pixmap)

            if seams_pixmap_item:
                self.processed_with_seams_scene.setSceneRect(seams_pixmap_item.boundingRect())
            self.ui.processedImageWithSeamsView.fitInView(self.processed_with_seams_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            
            # Store the seams pixmap
            self.seams_pixmap = pixmap

    def convert_cv_to_qimage(self, cv_image):
        """Convert OpenCV image to QImage"""
        if cv_image is None:
            return None
            
        try:
            # Convert BGR to RGB for color images
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image
            
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w if len(rgb_image.shape) == 3 else w
            
            # Create QImage
            if len(rgb_image.shape) == 3:
                q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            else:
                q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            
            return q_img.copy()  # Make a copy to avoid memory issues
        except Exception as e:
            print(f"Error converting CV image: {e}")
            return None

    def has_processed_image(self):
        """Check if a processed image exists"""
        return hasattr(self, 'processed_pixmap') and self.processed_pixmap is not None

    def set_seam_visualizations(self, visualizations: Dict):
        """Set seam visualizations for viewing"""
        if visualizations and isinstance(visualizations, dict):
            # Store all three visualization types
            self.seam_visualizations = {
                'all': visualizations.get('all'),  # Combined view
                'added': visualizations.get('added'),  # Added only
                'removed': visualizations.get('removed') # Removed only
            }
            
            # Update combo box options
            self.ui.seamsModeComboBox.clear()
            self.ui.seamsModeComboBox.addItems(["All Seams", "Added Seams", "Removed Seams"])
            self.ui.seamsModeComboBox.setCurrentText("All Seams")
            self.current_seams_mode = "All Seams"
            
            # Update button states
            self.update_button_states()
        else:
            self.seam_visualizations = None
            self.update_button_states()

    def enable_tool_buttons(self, enabled):
        """Enable or disable tool buttons based on image availability"""
        self.ui.resizeBtn.setEnabled(enabled)
        self.ui.filtersBtn.setEnabled(enabled)
        self.ui.obj_detectionBtn.setEnabled(enabled)
        
        # Update button states
        self.update_button_states()

    def update_timing_label(self, timing_info: Dict):
        """Update timing display label"""
        if timing_info:
            self.timing_info = timing_info
            self.update_timing_display()
        else:
            self.timing_info = None
            self.ui.timingLabel.setText("")
            self.ui.seamsTimingLabel.setText("")

    def update_timing_display(self):
        """Update timing display based on current page"""
        if not self.timing_info:
            self.ui.timingLabel.setText("")
            self.ui.seamsTimingLabel.setText("")
            return
        
        # Format timing information
        algorithm_time = self.timing_info.get('algorithm', 0.0)
        algorithm_name = self.timing_info.get('algorithm_name', 'Algorithm')
        
        timing_text = f"{algorithm_name}: {algorithm_time:.3f}s"
        
        # Update appropriate timing label based on current page
        if self.current_page_index == 1:  # Processed without seams page
            self.ui.timingLabel.setText(timing_text)
            self.ui.seamsTimingLabel.setText("")
        elif self.current_page_index == 2:  # Processed with seams page
            self.ui.seamsTimingLabel.setText(timing_text)
            self.ui.timingLabel.setText("")
        else:  # Original page (0) or other
            self.ui.timingLabel.setText("")
            self.ui.seamsTimingLabel.setText("")

    def update_dimension_labels(self):
        """Update dimension labels for all views"""
        if hasattr(self, 'original_pixmap') and self.original_pixmap is not None:
            w = self.original_pixmap.width()
            h = self.original_pixmap.height()
            self.ui.imageDimensionsLabel.setText(f"Width: {w} × Height: {h}")
        
        if hasattr(self, 'processed_pixmap') and self.processed_pixmap is not None:
            w = self.processed_pixmap.width()
            h = self.processed_pixmap.height()
            self.ui.processedDimensionsLabel.setText(f"Width: {w} × Height: {h}")
        
        if hasattr(self, 'seams_pixmap') and self.seams_pixmap is not None:
            w = self.seams_pixmap.width()
            h = self.seams_pixmap.height()
            self.ui.processedWithSeamsDimensionsLabel.setText(f"Width: {w} × Height: {h}")

    def reset_processed_state(self):
        """Reset processed image state"""
        self.processed_pixmap = None
        self.seams_pixmap = None
        self.seam_visualizations = None
        self.timing_info = None
        
        # Clear processed scenes
        self.processed_scene.clear()
        self.processed_with_seams_scene.clear()
        
        # Reset seam view state
        self.seams_view_active = False
        self.current_seams_mode = "All Seams"
        
        # Update UI
        self.ui.seamsViewBtn.setEnabled(False)
        self.ui.seamsViewBtn.setVisible(False)
        self.ui.seamsModeComboBox.setVisible(False)
        self.ui.seamsViewBtn.setText("View Seams")
        
        # Show placeholder for processed views
        font = QtGui.QFont()
        font.setPointSize(12)
        
        processed_text_item = self.processed_scene.addText("Processed image will appear here")
        processed_text_item.setDefaultTextColor(QtGui.QColor(108, 117, 125))
        processed_text_item.setFont(font)
        
        seams_text_item = self.processed_with_seams_scene.addText("Seams visualization will appear here")
        seams_text_item.setDefaultTextColor(QtGui.QColor(108, 117, 125))
        seams_text_item.setFont(font)
        
        # Clear labels
        self.ui.processedDimensionsLabel.setText("")
        self.ui.processedWithSeamsDimensionsLabel.setText("")
        self.ui.timingLabel.setText("")
        self.ui.seamsTimingLabel.setText("")
        
        # Update button states
        self.update_button_states()
        self.update_toggle_button_text()


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


class ComparativeAnalysisDialog(QtWidgets.QDialog):
    """Dialog for comparing Hubble 001 vs Hubble 002 algorithms"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Load UI if available, otherwise create dynamically
        self.ui = Ui_ComparisonDialog()
        self.ui.setupUi(self)
        
        self.setWindowTitle("Algorithm Comparison - Hubble 001 vs Hubble 002")
        self.setMinimumSize(1000, 700)
        
        # Store results
        self.comparison_results = {}
        
        # Connect signals
        self.ui.closeBtn.clicked.connect(self.accept)
        if hasattr(self.ui, 'exportBtn'):
            self.ui.exportBtn.clicked.connect(self.export_report)
    
    def display_comparison_results(self, results: Dict):
        """Display comparative analysis results"""
        self.comparison_results = results
        
        # Clear previous content
        if hasattr(self.ui, 'hubble001Scene'):
            self.ui.hubble001Scene.clear()
        if hasattr(self.ui, 'hubble002Scene'):
            self.ui.hubble002Scene.clear()
        
        # Create scenes if they don't exist
        if not hasattr(self, 'hubble001_scene'):
            self.hubble001_scene = QtWidgets.QGraphicsScene()
            self.ui.hubble001View.setScene(self.hubble001_scene)
        if not hasattr(self, 'hubble002_scene'):
            self.hubble002_scene = QtWidgets.QGraphicsScene()
            self.ui.hubble002View.setScene(self.hubble002_scene)
        
        # Display images and metrics
        report_text = "=== ALGORITHM COMPARISON RESULTS ===\n\n"
        
        for algo_name, data in results.items():
            # Convert image to QPixmap and display
            result_image = data.get('result_image')
            if result_image is not None:
                q_image = self.convert_cv_to_qimage(result_image)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    
                    if algo_name == "Hubble 001":
                        self.hubble001_scene.clear()
                        self.hubble001_scene.addPixmap(pixmap)
                        self.ui.hubble001View.fitInView(self.hubble001_scene.sceneRect(), 
                                                      QtCore.Qt.KeepAspectRatio)
                        
                        # Update time label
                        metrics = data.get('metrics', {})
                        exec_time = metrics.get('execution_time', 0)
                        self.ui.hubble001TimeLabel.setText(f"Time: {exec_time:.3f}s")
                    else:
                        self.hubble002_scene.clear()
                        self.hubble002_scene.addPixmap(pixmap)
                        self.ui.hubble002View.fitInView(self.hubble002_scene.sceneRect(), 
                                                      QtCore.Qt.KeepAspectRatio)
                        
                        # Update time label
                        metrics = data.get('metrics', {})
                        exec_time = metrics.get('execution_time', 0)
                        self.ui.hubble002TimeLabel.setText(f"Time: {exec_time:.3f}s")
        
        # Populate metrics table
        self.populate_metrics_table(results)
        
        # Generate analysis report
        report_text += self.generate_analysis_report(results)
        self.ui.analysisText.setText(report_text)
        
        # Determine overall winner
        self.determine_overall_winner(results)
    
    def populate_metrics_table(self, results: Dict):
        """Populate the metrics table with comparison data"""
        if not results:
            return
        
        # Define metrics to compare
        metrics_to_compare = [
            ("Execution Time (s)", "execution_time", True),  # Lower is better
            ("Memory Usage (MB)", "memory_usage_mb", True),  # Lower is better
            ("PSNR (dB)", "image_quality.psnr", False),     # Higher is better
            ("SSIM", "image_quality.ssim", False),          # Higher is better
            ("Seams/sec", "seams_per_second", False)        # Higher is better
        ]
        
        self.ui.metricsTable.setRowCount(len(metrics_to_compare))
        
        for row, (metric_name, metric_path, lower_is_better) in enumerate(metrics_to_compare):
            # Get values for both algorithms
            hubble001_val = self.get_nested_value(results.get("Hubble 001", {}), metric_path, 0)
            hubble002_val = self.get_nested_value(results.get("Hubble 002", {}), metric_path, 0)
            
            # Calculate difference and winner
            if hubble001_val is not None and hubble002_val is not None:
                diff = hubble001_val - hubble002_val
                
                if lower_is_better:
                    winner = "Hubble 001" if hubble001_val < hubble002_val else "Hubble 002"
                else:
                    winner = "Hubble 001" if hubble001_val > hubble002_val else "Hubble 002"
                
                # Calculate percentage change
                if hubble002_val != 0:
                    pct_change = (diff / abs(hubble002_val)) * 100
                else:
                    pct_change = 0
            else:
                diff = 0
                winner = "N/A"
                pct_change = 0
            
            # Populate table
            self.ui.metricsTable.setItem(row, 0, QtWidgets.QTableWidgetItem(metric_name))
            self.ui.metricsTable.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{hubble001_val:.3f}"))
            self.ui.metricsTable.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{hubble002_val:.3f}"))
            self.ui.metricsTable.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{diff:+.3f}"))
            
            winner_item = QtWidgets.QTableWidgetItem(winner)
            if winner == "Hubble 001":
                winner_item.setForeground(QtGui.QColor(66, 153, 225))  # Blue
            elif winner == "Hubble 002":
                winner_item.setForeground(QtGui.QColor(159, 122, 234))  # Purple
            self.ui.metricsTable.setItem(row, 4, winner_item)
            
            pct_item = QtWidgets.QTableWidgetItem(f"{pct_change:+.1f}%")
            if pct_change > 0:
                pct_item.setForeground(QtGui.QColor(56, 161, 105))  # Green for positive
            elif pct_change < 0:
                pct_item.setForeground(QtGui.QColor(229, 62, 62))  # Red for negative
            self.ui.metricsTable.setItem(row, 5, pct_item)
    
    def get_nested_value(self, obj, path, default=None):
        """Get nested value from dictionary using dot notation"""
        keys = path.split('.')
        value = obj
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value if value is not None else default
    
    def generate_analysis_report(self, results: Dict) -> str:
        """Generate detailed analysis report"""
        if not results:
            return "No comparison data available."
        
        report = ""
        
        for algo_name, data in results.items():
            metrics = data.get('metrics', {})
            seam_info = data.get('seam_info', {})
            
            report += f"\n{algo_name}:\n"
            report += "-" * 40 + "\n"
            
            # Performance metrics
            report += f"Performance Metrics:\n"
            report += f"  • Execution Time: {metrics.get('execution_time', 0):.3f}s\n"
            report += f"  • Memory Usage: {metrics.get('memory_usage_mb', 0):.1f} MB\n"
            
            # Seam statistics
            seam_stats = metrics.get('seam_count', {})
            if seam_stats:
                report += f"  • Seams Processed:\n"
                report += f"    - Total: {seam_stats.get('total', 0)}\n"
                report += f"    - Vertical Removed: {seam_stats.get('vertical_removed', 0)}\n"
                report += f"    - Vertical Inserted: {seam_stats.get('vertical_inserted', 0)}\n"
                report += f"    - Horizontal Removed: {seam_stats.get('horizontal_removed', 0)}\n"
                report += f"    - Horizontal Inserted: {seam_stats.get('horizontal_inserted', 0)}\n"
            
            # Image quality
            quality = metrics.get('image_quality', {})
            if quality:
                report += f"  • Image Quality:\n"
                report += f"    - PSNR: {quality.get('psnr', 0):.2f} dB\n"
                report += f"    - SSIM: {quality.get('ssim', 0):.3f}\n"
                report += f"    - Combined Score: {quality.get('combined_score', 0):.3f}\n"
            
            # Timing details if available
            if seam_info and 'timing' in seam_info:
                timing = seam_info['timing']
                report += f"  • Timing Breakdown:\n"
                report += f"    - Algorithm: {timing.get('algorithm', 0):.3f}s\n"
                report += f"    - Total: {timing.get('total', 0):.3f}s\n"
            
            report += "\n"
        
        return report
    
    def determine_overall_winner(self, results: Dict):
        """Determine overall winner based on multiple metrics"""
        if not results or len(results) < 2:
            self.ui.winnerLabel.setText("Overall Winner: Insufficient Data")
            return
        
        hubble001 = results.get("Hubble 001", {}).get('metrics', {})
        hubble002 = results.get("Hubble 002", {}).get('metrics', {})
        
        # Score system
        hubble001_score = 0
        hubble002_score = 0
        
        # Compare execution time (lower is better)
        time1 = hubble001.get('execution_time', float('inf'))
        time2 = hubble002.get('execution_time', float('inf'))
        if time1 < time2:
            hubble001_score += 2
        elif time2 < time1:
            hubble002_score += 2
        
        # Compare memory usage (lower is better)
        mem1 = hubble001.get('memory_usage_mb', float('inf'))
        mem2 = hubble002.get('memory_usage_mb', float('inf'))
        if mem1 < mem2:
            hubble001_score += 1
        elif mem2 < mem1:
            hubble002_score += 1
        
        # Compare PSNR (higher is better)
        psnr1 = hubble001.get('image_quality', {}).get('psnr', 0)
        psnr2 = hubble002.get('image_quality', {}).get('psnr', 0)
        if psnr1 > psnr2:
            hubble001_score += 1
        elif psnr2 > psnr1:
            hubble002_score += 1
        
        # Compare SSIM (higher is better)
        ssim1 = hubble001.get('image_quality', {}).get('ssim', 0)
        ssim2 = hubble002.get('image_quality', {}).get('ssim', 0)
        if ssim1 > ssim2:
            hubble001_score += 1
        elif ssim2 > ssim1:
            hubble002_score += 1
        
        # Determine winner
        if hubble001_score > hubble002_score:
            winner = "Hubble 001 (Sequential)"
            color = "#4299e1"  # Blue
        elif hubble002_score > hubble001_score:
            winner = "Hubble 002 (Bulk)"
            color = "#9f7aea"  # Purple
        else:
            winner = "Tie"
            color = "#d69e2e"  # Yellow
        
        self.ui.winnerLabel.setText(f"Overall Winner: {winner}")
        self.ui.winnerLabel.setStyleSheet(f"""
            font-size: 13pt; font-weight: 800; color: #2d3748; 
            padding: 12px; background-color: #f0fff4;
            border-radius: 8px; border: 3px solid {color};
        """)
    
    def convert_cv_to_qimage(self, cv_image):
        """Convert OpenCV image to QImage"""
        if cv_image is None:
            return None
            
        try:
            # Convert BGR to RGB for color images
            if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv_image
            
            h, w = rgb_image.shape[:2]
            bytes_per_line = 3 * w if len(rgb_image.shape) == 3 else w
            
            # Create QImage
            if len(rgb_image.shape) == 3:
                q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            else:
                q_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            
            return q_img.copy()  # Make a copy to avoid memory issues
        except Exception as e:
            print(f"Error converting CV image: {e}")
            return None
    
    def export_report(self):
        """Export comparison report to file"""
        if not self.comparison_results:
            QtWidgets.QMessageBox.warning(self, "No Data", "No comparison data to export.")
            return
        
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Comparison Report",
            "algorithm_comparison_report.txt",
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("=== ALGORITHM COMPARISON REPORT ===\n\n")
                    f.write("Generated: " + QtCore.QDateTime.currentDateTime().toString() + "\n\n")
                    
                    # Write table data
                    f.write("PERFORMANCE METRICS:\n")
                    f.write("-" * 80 + "\n")
                    
                    # Write table headers
                    headers = []
                    for col in range(self.ui.metricsTable.columnCount()):
                        header = self.ui.metricsTable.horizontalHeaderItem(col)
                        if header:
                            headers.append(header.text())
                    
                    f.write(" | ".join(headers) + "\n")
                    f.write("-" * 80 + "\n")
                    
                    # Write table rows
                    for row in range(self.ui.metricsTable.rowCount()):
                        row_data = []
                        for col in range(self.ui.metricsTable.columnCount()):
                            item = self.ui.metricsTable.item(row, col)
                            row_data.append(item.text() if item else "")
                        f.write(" | ".join(row_data) + "\n")
                    
                    f.write("\n\n")
                    
                    # Write analysis text
                    f.write("DETAILED ANALYSIS:\n")
                    f.write("-" * 80 + "\n")
                    f.write(self.ui.analysisText.toPlainText())
                    
                    # Write winner
                    f.write("\n\n" + self.ui.winnerLabel.text())
                
                QtWidgets.QMessageBox.information(self, "Export Successful", 
                                                f"Report exported to:\n{file_path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export Error", 
                                             f"Failed to export report: {str(e)}")