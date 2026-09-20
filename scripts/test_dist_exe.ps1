$exePath = "D:\Project-monitoring-model\dist\Fall_and_Stroke_Warning_System\Fall_and_Stroke_Warning_System.exe"
$logPath = "D:\Project-monitoring-model\dist\Fall_and_Stroke_Warning_System\app_log.txt"

Write-Host "[1] Starting Fall_and_Stroke_Warning_System.exe..."
$proc = Start-Process -FilePath $exePath -PassThru

Write-Host "[2] Waiting 6 seconds for startup..."
Start-Sleep -Seconds 6

Write-Host "[3] Checking app_log.txt output:"
if (Test-Path $logPath) {
    Get-Content $logPath -Tail 25
} else {
    Write-Host "app_log.txt not found yet"
}

Write-Host "[4] Probing http://localhost:8000/ ..."
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing -TimeoutSec 3
    Write-Host "HTTP Response Code:" $resp.StatusCode
} catch {
    Write-Host "HTTP probe failed:" $_.Exception.Message
}

Write-Host "[5] Sending shutdown request..."
try {
    $shutdown = Invoke-RestMethod -Uri "http://localhost:8000/api/system/shutdown" -Method Post -TimeoutSec 2
    Write-Host "Shutdown response:" $shutdown.message
} catch {
    Write-Host "Shutdown request failed, stopping process manually..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2
if ($proc.HasExited) {
    Write-Host "[SUCCESS] Process has exited cleanly!"
} else {
    Write-Host "[INFO] Stopping process..."
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
