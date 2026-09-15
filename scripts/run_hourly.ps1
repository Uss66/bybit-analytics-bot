# Hourly job for Task Scheduler: refresh live data, then run the paper
# trader in local-simulation mode (see testnet_trader.py's --mode simulate
# docstring for why simulate is the default - Bybit testnet account access
# never got resolved, see project_testnet_execution memory).
#
# Registered as a Windows scheduled task named "ByBitAnalitics-HourlyTrader".
# Logs to logs/hourly_<timestamp>.log for later review; logs/ is
# already covered by .gitignore's *.log pattern.

$ErrorActionPreference = "Continue"
$PSDefaultParameterValues['Out-File:Encoding'] = 'utf8'  # PowerShell's default (UTF-16LE) makes logs unreadable in plain text tools
$root = "C:\TEST_AI\ByBit_Analitics"
$python = "$root\.venv\Scripts\python.exe"
$logDir = "$root\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$logFile = "$logDir\hourly_$(Get-Date -Format 'yyyy-MM-dd').log"

Set-Location "$root\scripts"

"[$(Get-Date -Format o)] --- refresh_live_data.py ---" | Out-File -Append $logFile
& $python refresh_live_data.py *>&1 | Out-File -Append $logFile

"[$(Get-Date -Format o)] --- testnet_trader.py --mode simulate ---" | Out-File -Append $logFile
& $python testnet_trader.py --mode simulate *>&1 | Out-File -Append $logFile

"[$(Get-Date -Format o)] --- done ---" | Out-File -Append $logFile
