$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        throw 'Project environment is missing. Follow the README setup instructions first.'
    }
    & '.venv/Scripts/python.exe' -m camera_system.app --port 8780
} finally {
    Pop-Location
}
