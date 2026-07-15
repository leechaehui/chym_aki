# CHYM-AKI: ensure PostgreSQL is available before the setup DB step.
# - If 5432 already answers, pass immediately (no prompt).
# - If the service is stopped, start it; if not elevated, elevate once via UAC.
# - Wait until the port accepts connections, then exit 0 (ok) / 1 (failed).
# (ASCII-only output so it stays readable on any console code page.)
param(
    [string]$Service = 'postgresql-x64-16',
    [int]$Port = 5432,
    [int]$TimeoutSec = 30
)

function Test-PgPort {
    (Test-NetConnection 127.0.0.1 -Port $Port -WarningAction SilentlyContinue).TcpTestSucceeded
}

if (Test-PgPort) {
    Write-Host "[PG] Already accepting on $Port - OK."
    exit 0
}

$svc = Get-Service -Name $Service -ErrorAction SilentlyContinue
if (-not $svc) {
    Write-Warning "[PG] Service '$Service' not found. Check PostgreSQL install / service name."
    exit 1
}

Write-Host "[PG] Starting service '$Service'..."
try {
    Start-Service -Name $Service -ErrorAction Stop
} catch {
    Write-Host "[PG] Elevation required - click 'Yes' on the UAC prompt..."
    try {
        Start-Process powershell -Verb RunAs -Wait -ArgumentList `
            '-NoProfile','-Command',"Start-Service $Service" -ErrorAction Stop
    } catch {
        Write-Warning "[PG] Elevated start failed (denied?): $($_.Exception.Message)"
        exit 1
    }
}

for ($i = 0; $i -lt $TimeoutSec; $i++) {
    if (Test-PgPort) {
        Write-Host "[PG] Ready - 127.0.0.1:$Port responding."
        exit 0
    }
    Start-Sleep -Seconds 1
}

Write-Warning "[PG] Port $Port not open within ${TimeoutSec}s. Check for a stale postmaster.pid in the data folder."
exit 1
