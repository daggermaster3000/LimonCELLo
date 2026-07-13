$destination = "C:\Users\qfavey\Documents\Thomaso\training_dataset\Data"
$files = Get-Content "C:\Users\qfavey\Documents\Thomaso\training_dataset\files.txt"

foreach ($file in $files) {
    if (Test-Path $file) {
        Copy-Item -LiteralPath $file -Destination $destination -Force
    } else {
        Write-Warning "Missing: $file"
    }
}