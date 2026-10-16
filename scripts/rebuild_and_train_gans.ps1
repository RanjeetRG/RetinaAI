# Rebuild the model-input arrays after a preprocessing change, then (re)start GAN training (Windows side).
#
#     powershell -ExecutionPolicy Bypass -File scripts\rebuild_and_train_gans.ps1
#
# 1. GAN arrays (needed before training), 2. start the four-transition chain in WSL with keep-awake,
# 3. CNN arrays on the CPU while the GPU trains. Logs: outputs\rebuild_arrays.log, outputs\train_gans.log
#
# Python runs through cmd so its stderr (progress bars) is logged as plain text: in Windows PowerShell 5.1,
# redirecting a native program's stderr turns each line into an error record, which stops the script.
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "outputs\rebuild_arrays.log"
Set-Location $root

function Note($text) { Add-Content -Path $log -Value "=== $text $(Get-Date -Format s)" -Encoding ascii }

Set-Content -Path $log -Value "" -Encoding ascii
Note "GAN arrays"
cmd /c "`"$python`" -u -m gan.data >> `"$log`" 2>&1"
if ($LASTEXITCODE -ne 0) { Note "gan.data failed with exit code $LASTEXITCODE"; exit $LASTEXITCODE }

$wslRoot = "/mnt/" + $root.Substring(0, 1).ToLower() + $root.Substring(2).Replace("\", "/")  # D:\x\y -> /mnt/d/x/y
$train = Start-Process -FilePath "wsl.exe" -ArgumentList "-d","Ubuntu","--","bash","$wslRoot/scripts/wsl_run.sh","train_gans","bash","scripts/train_gans.sh" -WindowStyle Hidden -PassThru
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",(Join-Path $PSScriptRoot "keep_awake.ps1"),"-ProcessId",$train.Id -WindowStyle Hidden | Out-Null
Note "GAN training started, wsl PID $($train.Id)"

Note "CNN arrays"
cmd /c "`"$python`" -u -m cnn.data >> `"$log`" 2>&1"
Note "done, cnn.data exit code $LASTEXITCODE"
