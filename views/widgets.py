"""
views/widgets.py
Widgets for ImageLab main window and tools
"""

from typing import Dict, Union
from PyQt5 import QtWidgets, QtCore, QtGui
import cv2

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

        self.ui = Ui_ImageLab()
        self.ui.setupUi(self)

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
            print("all seams_img", seams_img)
            if seams_img is not None:
                q_image = self.convert_cv_to_qimage(seams_img)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.display_processed_with_seams_image(pixmap)

    def show_removed_seams(self):
        """Show processed image with removed seams highlighted"""
        if self.seam_visualizations is not None and 'removed' in self.seam_visualizations:
            seams_img = self.seam_visualizations['removed']
            print("removed seams_img", seams_img)
            if seams_img is not None:
                q_image = self.convert_cv_to_qimage(seams_img)
                if q_image:
                    pixmap = QtGui.QPixmap.fromImage(q_image)
                    self.display_processed_with_seams_image(pixmap)

    def show_added_seams(self):
        """Show processed image with added seams highlighted"""
        if self.seam_visualizations is not None and 'added' in self.seam_visualizations:
            seams_img = self.seam_visualizations['added']
            print("added seams_img", seams_img)
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
