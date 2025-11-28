#!/bin/bash

# Default values
UI_PATH="./uidesigns"
HELP=false

# Function to display help
show_help() {
    echo "Usage: $0 [-u|--uipath <path>] [-h|--help]"
    echo "  -u, --uipath: Path to directory containing .ui files (default: ./uidesigns)"
    echo "  -h, --help: Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--uipath)
            UI_PATH="$2"
            shift 2
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    show_help
    exit 0
fi

# Check if pyuic5 is available
if ! command -v pyuic5 &> /dev/null; then
    echo "Error: pyuic5 not found. Please ensure PyQt5 is installed and in your PATH."
    exit 1
fi

# Check if pyrcc5 is available
if ! command -v pyrcc5 &> /dev/null; then
    echo "Error: pyrcc5 not found. Please ensure PyQt5 is installed and in your PATH."
    exit 1
fi

# Check if UI directory exists
if [ ! -d "$UI_PATH" ]; then
    echo "Error: Directory $UI_PATH does not exist."
    exit 1
fi

# Get all .ui files in the directory
ui_files=("$UI_PATH"/*.ui)

if [ ${#ui_files[@]} -eq 0 ] || [ ! -f "${ui_files[0]}" ]; then
    echo "No .ui files found in $UI_PATH"
    exit 0
fi

echo "Converting .ui files in $UI_PATH..." 

# Create resources directory if it doesn't exist
mkdir -p "./resources"

# Convert resource file first (if it exists)
if [ -f "./resources/resources.qrc" ]; then
    echo "Converting resource file..."
    pyrcc5 -o "./resources/resources_rc.py" "./resources/resources.qrc"
fi

# Convert each .ui file
for ui_file in "${ui_files[@]}"; do
    if [ ! -f "$ui_file" ]; then
        continue
    fi
    
    filename=$(basename -- "$ui_file")
    base_name="${filename%.*}"
    output_file="$UI_PATH/${base_name}.py"
    
    echo "Converting: $filename -> $(basename "$output_file")"
    
    if pyrcc5 -o "./resources/resources_rc.py" "./resources/resources.qrc" && \
       pyuic5 --import-from=resources -o "$output_file" "$ui_file"; then
        echo "✓ Successfully converted: $filename"
    else
        echo "✗ Failed to convert: $filename"
        exit 1
    fi
done

echo "Conversion completed!"