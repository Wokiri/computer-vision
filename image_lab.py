from pathlib import Path
import numpy as np
import re
from uidesigns.main_window_gui import Ui_ImageLab
from utilities.processing import ImageProcessor
from views.widgets import FilterWidget, ObjectDetectionWidget, ResizeWidget

from PyQt5 import QtCore, QtGui, QtWidgets


class ImageLab(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_ImageLab()
        self.ui.setupUi(self)

        self.resize_widget = ResizeWidget(self)
        self.filter_widget = FilterWidget(self)
        self.object_detection_widget = ObjectDetectionWidget(self)

        self.initialize_ui()

        self.image_processor = None

        # Track the currently active tool
        self.current_tool = None
        
        # Track current image
        self.current_image = None
        self.current_image_path = None
        self.current_pixmap_item = None
        
        # Zoom tracking
        self.current_zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

        # Define aspect ratio presets
        self.aspect_ratios = {
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

        self.ui.resizeBtn.clicked.connect(self.processing_selection)
        self.ui.obj_detectionBtn.clicked.connect(self.processing_selection)
        self.ui.filtersBtn.clicked.connect(self.processing_selection)

        # Connect UploadButton
        self.ui.UploadButton.clicked.connect(self.select_and_display_image)
        
        # Connect Zoom ComboBox
        self.ui.selectStandardZoomComboBox.currentTextChanged.connect(self.apply_standard_zoom)

        self.ui.zoomInButton.clicked.connect(lambda: self.apply_zoom_in(5))
        self.ui.zoomOutButton.clicked.connect(lambda: self.apply_zoom_out(5))

        # Connect signals
        self.resize_widget.ui.width_resize_lineEdit.textChanged.connect(self.on_width_changed)
        self.resize_widget.ui.height_resize_lineEdit.textChanged.connect(self.on_height_changed)
        self.resize_widget.ui.resize_image_comboBox.currentTextChanged.connect(self.on_aspect_ratio_changed)

    def initialize_ui(self):
        """Initialize UI settings for QGraphicsView and zoom controls"""
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
        
        # Enable mouse wheel zoom
        self.ui.ImagePreview.wheelEvent = self.graphics_view_wheel_event

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

    def apply_standard_zoom(self, zoom_option):
        """Apply standard zoom level based on combo box selection"""
        if not self.current_pixmap_item:
            return
            
        if zoom_option == "Fit to View":
            self.zoom_fit_to_view()
        elif zoom_option == "Actual Size":
            self.zoom_actual_size()
        elif zoom_option.endswith('%'):
            self.zoom_by_percentage(zoom_option)
        elif "custom" in zoom_option.lower():
            self.show_custom_zoom_dialog()

    def zoom_fit_to_view(self):
        """Zoom to fit the entire image in the view"""
        if self.current_pixmap_item:
            self.ui.ImagePreview.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            self.current_zoom_level = self.calculate_current_zoom_level()
            self.update_zoom_combo_box_display()

    def zoom_actual_size(self):
        """Zoom to 100% (actual pixel size)"""
        if self.current_pixmap_item:
            self.apply_zoom_level(1.0)

    def zoom_by_percentage(self, percentage_text):
        """Zoom by percentage value"""
        try:
            # Extract percentage number from text (remove % sign)
            percentage = float(percentage_text.rstrip('%'))
            zoom_level = percentage / 100.0
            
            if self.min_zoom <= zoom_level <= self.max_zoom:
                self.apply_zoom_level(zoom_level)
                
        except ValueError:
            print(f"Invalid percentage format: {percentage_text}")

    def apply_zoom_in(self, zoom_factor=5, checked=None):
        if self.current_pixmap_item:
            current_zoom = self.calculate_current_zoom_level()
            
            # Convert to percentage for easier calculation
            current_percent = current_zoom * 100
            
            # Find the next multiple of zoom_factor
            next_multiple = ((current_percent + zoom_factor - 1) // zoom_factor) * zoom_factor
            
            # Ensure we're moving forward by at least one step
            if next_multiple <= current_percent:
                next_multiple += zoom_factor
                
            zoom_level = next_multiple / 100.0
            
            zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
            self.apply_zoom_level(zoom_level)

    def apply_zoom_out(self, zoom_factor=5, checked=None):
        if self.current_pixmap_item:
            current_zoom = self.calculate_current_zoom_level()
            
            # Convert to percentage for easier calculation
            current_percent = current_zoom * 100
            
            # Find the previous multiple of zoom_factor
            prev_multiple = ((current_percent - 1) // zoom_factor) * zoom_factor
            
            # Ensure we're moving backward by at least one step
            if prev_multiple >= current_percent:
                prev_multiple -= zoom_factor
                
            zoom_level = prev_multiple / 100.0
            
            zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
            self.apply_zoom_level(zoom_level)

    def apply_zoom_level(self, zoom_level):
        """Apply specific zoom level"""
        if not self.current_pixmap_item:
            return
            
        # Clamp zoom level to min/max values
        zoom_level = max(self.min_zoom, min(self.max_zoom, zoom_level))
        
        # Reset transformation
        transform = QtGui.QTransform()
        # Apply zoom
        transform.scale(zoom_level, zoom_level)
        self.ui.ImagePreview.setTransform(transform)
        
        self.current_zoom_level = zoom_level
        self.update_zoom_combo_box_display()

    def calculate_current_zoom_level(self):
        """Calculate current zoom level based on view transformation"""
        if not self.current_pixmap_item:
            return 1.0
            
        # Get the transform and extract scale
        transform = self.ui.ImagePreview.transform()
        return transform.m11()  # Horizontal scale component

    def get_zoom_combo_box_items(self):
        combo = self.ui.selectStandardZoomComboBox
        return [combo.itemText(i) for i in range(combo.count())]

    def update_zoom_combo_box_display(self):
        """Update combo box to reflect current zoom level, without triggering events"""
        # Block signals to prevent infinite loop
        self.ui.selectStandardZoomComboBox.blockSignals(True)
        
        # Calculate percentage
        percentage = int(self.current_zoom_level * 100)
        
        # Check if it matches any standard option
        standard_option = f"{percentage}%"
        index = self.ui.selectStandardZoomComboBox.findText(standard_option)
        
        if index >= 0:
            self.ui.selectStandardZoomComboBox.setCurrentText(standard_option)
        else:
            custom_index = self.ui.selectStandardZoomComboBox.findText(
                "^Custom", QtCore.Qt.MatchRegularExpression
            )
            if custom_index >= 0:
                # Use the existing custom item
                self.ui.selectStandardZoomComboBox.setCurrentIndex(custom_index)
                # Update it to show the current percentage
                self.ui.selectStandardZoomComboBox.setItemText(
                    custom_index, 
                    f"Custom ({percentage}%)"
                )
            else:
                # Fallback: just set text directly
                self.ui.selectStandardZoomComboBox.setCurrentText(f"Custom ({percentage}%)")
        
        # Restore signals
        self.ui.selectStandardZoomComboBox.blockSignals(False)

    def show_custom_zoom_dialog(self):
        """Show dialog for custom zoom input"""
        if not self.current_pixmap_item:
            return
            
        current_percentage = int(self.current_zoom_level * 100)
        
        zoom_value, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Custom Zoom",
            "Enter zoom percentage:",
            value=current_percentage,
            min=10,   # 10%
            max=1000, # 1000%
            step=10
        )
        
        if ok:
            zoom_level = zoom_value / 100.0
            self.apply_zoom_level(zoom_level)

    def graphics_view_wheel_event(self, event):
        """Handle mouse wheel events for zooming"""
        if event.modifiers() & QtCore.Qt.ControlModifier:
            # Zoom with Ctrl + Mouse Wheel
            zoom_factor = 1.15  # 15% zoom per step
            if event.angleDelta().y() > 0:
                # Zoom in
                new_zoom = self.current_zoom_level * zoom_factor
            else:
                # Zoom out
                new_zoom = self.current_zoom_level / zoom_factor
            
            # Clamp zoom level
            new_zoom = max(self.min_zoom, min(self.max_zoom, new_zoom))
            
            # Apply zoom
            self.apply_zoom_level(new_zoom)
            
            # Update combo box
            self.update_zoom_combo_box_display()
            
            event.accept()
        else:
            # Default wheel behavior (scroll)
            QtWidgets.QGraphicsView.wheelEvent(self.ui.ImagePreview, event)

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

    def select_and_display_image(self):
        """Select an image file and display it in the preview"""
        # Supported image formats
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp);;All Files (*)"
        
        # Open file dialog
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose an Image",
            str(Path.home()),  # Start from home directory
            file_filter
        )
        
        if file_path:
            self.load_and_display_image(file_path)

    def load_and_display_image(self, file_path):
        """Load image from file path and display in preview"""
        try:
            # Clear previous image
            self.clear_image_preview()
            
            # Load image
            pixmap = QtGui.QPixmap(file_path)
            
            if pixmap.isNull():
                QtWidgets.QMessageBox.warning(self, "Error", "Unable to load image file.")
                return
            
            # Store image data
            self.current_image_path = file_path
            self.current_image = pixmap
            
            # Display image in QGraphicsView
            self.display_image_in_preview(pixmap)
            
            # Reset to "Fit to View" when new image is loaded
            self.ui.selectStandardZoomComboBox.setCurrentText("Fit to View")
            self.zoom_fit_to_view()
            
            # Update window title with filename
            filename = Path(file_path).name
            self.setWindowTitle(f"ImageLab - {filename}")
            
            # Enable tool buttons if they were disabled
            self.enable_tool_buttons(True)

            self.image_processor = ImageProcessor(file_path)
            
            if self.image_processor.is_image_loaded():
                self.statusBar().showMessage(f"Successfully loaded image: {file_path}", 3000)

                w, h = self.image_processor.get_image_dimensions()

                # Store original dimensions and aspect ratio
                self.original_width = w
                self.original_height = h
                self.aspect_ratio = w / h if h != 0 else 1

                self.aspect_ratios.update({
                    "Original": self.aspect_ratio
                })

                # Set initial values
                self.ui.imageDimensionsLabel.setText(f"Width: {w} × Height: {h}")
                self.resize_widget.ui.width_resize_lineEdit.setText(str(w))
                self.resize_widget.ui.height_resize_lineEdit.setText(str(h))
            
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def on_aspect_ratio_changed(self, ratio_name):
        """When aspect ratio selection changes"""
        if ratio_name == "Custom":
            return  # Don't auto-adjust in custom mode
        
        aspect_ratio = self.aspect_ratios[ratio_name]
        
        # Get current width and calculate new height
        try:
            current_width = int(self.resize_widget.ui.width_resize_lineEdit.text())
            new_height = int(round(current_width / aspect_ratio))
            
            # Block signals to prevent recursive calls
            self.resize_widget.ui.height_resize_lineEdit.blockSignals(True)
            self.resize_widget.ui.height_resize_lineEdit.setText(str(new_height))
            self.resize_widget.ui.height_resize_lineEdit.blockSignals(False)
            
        except ValueError:
            pass

    def on_width_changed(self, width_text):
        """When width line edit changes"""
        if self.resize_widget.ui.resize_image_comboBox.currentText() == "Custom":
            return  # Don't auto-adjust in custom mode
        
        try:
            width = int(width_text)
            ratio_name = self.resize_widget.ui.resize_image_comboBox.currentText()
            aspect_ratio = self.aspect_ratios[ratio_name]
            
            new_height = int(round(width / aspect_ratio))
            
            # Block signals to prevent recursive calls
            self.resize_widget.ui.height_resize_lineEdit.blockSignals(True)
            self.resize_widget.ui.height_resize_lineEdit.setText(str(new_height))
            self.resize_widget.ui.height_resize_lineEdit.blockSignals(False)
            
        except ValueError:
            pass  # Invalid input

    def on_height_changed(self, height_text):
        """When height line edit changes"""
        if self.resize_widget.ui.resize_image_comboBox.currentText() == "Custom":
            return  # Don't auto-adjust in custom mode
        
        try:
            height = int(height_text)
            ratio_name = self.resize_widget.ui.resize_image_comboBox.currentText()
            aspect_ratio = self.aspect_ratios[ratio_name]
            
            new_width = int(round(height * aspect_ratio))
            
            # Block signals to prevent recursive calls
            self.resize_widget.ui.width_resize_lineEdit.blockSignals(True)
            self.resize_widget.ui.width_resize_lineEdit.setText(str(new_width))
            self.resize_widget.ui.width_resize_lineEdit.blockSignals(False)
            
        except ValueError:
            pass

    def display_image_in_preview(self, pixmap):
        """Display pixmap in QGraphicsView with proper scaling"""
        # Clear the scene
        self.scene.clear()
        
        # Add pixmap to scene
        self.current_pixmap_item = self.scene.addPixmap(pixmap)
        
        # Set the scene rect to match the pixmap
        self.scene.setSceneRect(self.current_pixmap_item.boundingRect())

    def clear_image_preview(self):
        """Clear the image preview and reset to placeholder state"""
        self.scene.clear()
        self.current_image = None
        self.current_image_path = None
        self.current_pixmap_item = None
        self.current_zoom_level = 1.0
        
        # Reset to placeholder state
        self.show_placeholder_text()
        
        # Reset zoom combo box
        self.ui.selectStandardZoomComboBox.setCurrentText("Fit to View")
        
        # Reset window title
        self.setWindowTitle("ImageLab")
        
        # Disable tool buttons when no image is loaded
        self.enable_tool_buttons(False)

    def enable_tool_buttons(self, enabled):
        """Enable or disable tool buttons based on image availability"""
        self.ui.resizeBtn.setEnabled(enabled)
        self.ui.filtersBtn.setEnabled(enabled)
        self.ui.obj_detectionBtn.setEnabled(enabled)

    def resize_event(self, event):
        """Handle window resize to update image preview scaling"""
        super().resizeEvent(event)
        if self.current_pixmap_item and self.ui.selectStandardZoomComboBox.currentText() == "Fit to View":
            self.zoom_fit_to_view()

    def get_current_image(self):
        """Get the current loaded image as QPixmap"""
        return self.current_image

    def get_current_image_path(self):
        """Get the path of the current loaded image"""
        return self.current_image_path

    def has_image_loaded(self):
        """Check if an image is currently loaded"""
        return self.current_image is not None

    # Add Drag and Drop Support
    def drag_enter_event(self, event):
        """Handle drag enter event for file drops"""
        if event.mimeData().hasUrls():
            # Check if any of the dragged files are images
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self.is_image_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def drag_move_event(self, event):
        """Handle drag move event"""
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if self.is_image_file(file_path):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def drop_event(self, event):
        """Handle drop event for image files"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if self.is_image_file(file_path):
                self.load_and_display_image(file_path)
                event.acceptProposedAction()
                return
        event.ignore()

    def is_image_file(self, file_path):
        """Check if file is a supported image format"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}
        file_path = Path(file_path)
        return file_path.is_file() and file_path.suffix.lower() in image_extensions

    # Enable drag and drop for the main window
    def set_accept_drops(self, accept):
        """Override to enable drag and drop for the main window and graphics view"""
        super().setAcceptDrops(accept)
        self.ui.ImagePreview.setAcceptDrops(accept)

    def position_tool_widget(self, widget, side="right", margin=10):
        """Position tool widget with screen boundary checking"""
        main_window_geometry = self.geometry()
        screen_geometry = QtWidgets.QApplication.primaryScreen().availableGeometry()
        
        if side == "right":
            x = main_window_geometry.right() + margin
            y = main_window_geometry.top()
            
            # Check if widget would go off-screen to the right
            if x + widget.width() > screen_geometry.right():
                # Position to the left instead
                x = main_window_geometry.left() - widget.width() - margin
                
        elif side == "left":
            x = main_window_geometry.left() - widget.width() - margin
            y = main_window_geometry.top()
            
            # Check if widget would go off-screen to the left
            if x < screen_geometry.left():
                # Position to the right instead
                x = main_window_geometry.right() + margin
        
        # Ensure widget stays within screen vertically
        if y + widget.height() > screen_geometry.bottom():
            y = screen_geometry.bottom() - widget.height()
        if y < screen_geometry.top():
            y = screen_geometry.top()
            
        widget.move(x, y)

    def processing_selection(self):
        """Handle tool selection with image validation"""
        # Check if an image is loaded before showing tools
        if not self.has_image_loaded():
            QtWidgets.QMessageBox.information(
                self, 
                "No Image", 
                "Please select an image first before using tools."
            )
            return

        sender = self.sender()
        tool_to_show = None 

        if sender == self.ui.resizeBtn:
            tool_to_show = self.resize_widget
        elif sender == self.ui.filtersBtn:
            tool_to_show = self.filter_widget
        elif sender == self.ui.obj_detectionBtn:
            tool_to_show = self.object_detection_widget
        
        if tool_to_show:
            # Always hide the current tool first
            if self.current_tool:
                self.current_tool.hide()
            
            # If clicking the same tool button, toggle visibility
            if self.current_tool == tool_to_show:
                self.current_tool = None
            else:
                # Show the new tool and update current_tool
                self.position_tool_widget(tool_to_show, "right")
                tool_to_show.show()
                self.current_tool = tool_to_show


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    image_processor = ImageLab()
    
    # Enable drag and drop
    image_processor.setAcceptDrops(True)
    image_processor.ui.ImagePreview.setAcceptDrops(True)
    
    image_processor.show()
    sys.exit(app.exec_())