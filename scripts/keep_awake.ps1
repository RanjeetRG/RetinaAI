# Keep Windows from idle-sleeping until a given process exits (e.g. a long WSL training run).
# Changes no power settings: it holds an "execution state" request, like a video player does, and releases it
# when the process ends. Closing the laptop lid can still sleep the machine.
#
#     powershell -File scripts\keep_awake.ps1 -ProcessId <pid of the wsl.exe running training>
param([Parameter(Mandatory = $true)][int]$ProcessId)

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$ES_CONTINUOUS = [uint32]2147483648      # 0x80000000
$ES_SYSTEM_REQUIRED = [uint32]1

[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    Wait-Process -Id $ProcessId -ErrorAction SilentlyContinue
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
