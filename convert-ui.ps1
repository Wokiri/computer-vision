param(
    [string]$UiPath = ".\uidesigns",
    [switch]$Help
)

if ($Help) {
    Write-Host "Usage: .\convert-ui.ps1 [-UiPath <path>] [-Help]"
    Write-Host "  -UiPath: Path to directory containing .ui files (default: .\uidesigns)"
    Write-Host "  -Help: Show this help message"
    exit
}

# Check if pyuic5 is available
try {
    $null = Get-Command pyuic5 -ErrorAction Stop
}
catch {
    Write-Error "pyuic5 not found. Please ensure PyQt5 is installed and in your PATH."
    exit 1
}

# Check if UI directory exists
if (-not (Test-Path $UiPath)) {
    Write-Error "Directory $UiPath does not exist."
    exit 1
}

# Get all .ui files in the directory
$uiFiles = Get-ChildItem -Path $UiPath -Filter "*.ui"

if ($uiFiles.Count -eq 0) {
    Write-Host "No .ui files found in $UiPath"
    exit 0
}

Write-Host "Converting .ui files in $UiPath..." -ForegroundColor Green

foreach ($uiFile in $uiFiles) {
    $outputFile = Join-Path $UiPath ($uiFile.BaseName + ".py")
    
    Write-Host "Converting: $($uiFile.Name) -> $($outputFile)" -ForegroundColor Yellow
    
    try {
        pyrcc5 -o .\resources\resources_rc.py .\resources\resources.qrc
        pyuic5 -d --import-from=resources -o $outputFile $uiFile.FullName
        Write-Host "✓ Successfully converted: $($uiFile.Name)" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to convert: $($uiFile.Name)" -ForegroundColor Red
        Write-Error $_.Exception.Message
    }
}

Write-Host "Conversion completed!" -ForegroundColor Green