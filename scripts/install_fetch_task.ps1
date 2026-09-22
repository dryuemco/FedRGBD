<#
.SYNOPSIS
    Register the hourly FedRGBD result-fetch task on this desktop.

.DESCRIPTION
    Creates a Windows Task Scheduler task that runs scripts\fetch_results.ps1 every
    60 minutes as the current user.

    The task is registered with LogonType = Interactive, i.e. it runs only while you
    are logged on. That is deliberate: the SSH key lives in the Windows ssh-agent,
    whose keys are protected per user, and a task configured to "run whether the user
    is logged on or not" cannot reliably reach them. If you need fetching to continue
    while logged off, put a passphrase-less key on disk, set "identity_file" in the
    config, and re-register with -RunWhenLoggedOff.

    Nothing about this touches the Jetsons. The fetch itself is read-only on the node.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_fetch_task.ps1
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install_fetch_task.ps1 -IntervalMinutes 30
#>
[CmdletBinding()]
param(
    [string]$TaskName = 'FedRGBD-FetchResults',
    [int]$IntervalMinutes = 60,
    [switch]$RunWhenLoggedOff,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot 'fetch_results.ps1'
$Config = Join-Path $PSScriptRoot 'fetch_results.config.json'

if (-not (Test-Path $Script)) { throw "missing $Script" }
if (-not (Test-Path $Config)) {
    throw ("missing {0}`n  Copy scripts\fetch_results.config.example.json to that name and fill it in first." -f $Config)
}

$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    throw ("task '{0}' already exists. Re-run with -Force to replace it, or remove it with scripts\uninstall_fetch_task.ps1" -f $TaskName)
}
if ($existing) { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false }

$action = New-ScheduledTaskAction `
    -Execute 'powershell.exe' `
    -Argument ('-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "{0}"' -f $Script) `
    -WorkingDirectory $RepoRoot

# Start a couple of minutes out, then repeat forever.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)
try { $trigger.Repetition.Duration = 'P3650D' } catch { }   # ~10 years == "indefinitely"

# ExecutionTimeLimit is the backstop: even if ssh somehow blocks, the task is killed
# and the next hour's pass starts clean. IgnoreNew stops passes from piling up.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

if ($RunWhenLoggedOff) {
    Write-Warning 'Registering with LogonType S4U: the task will run while logged off, but it can only authenticate with an identity_file, not with the ssh-agent.'
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType S4U -RunLevel Limited
} else {
    $principal = New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
}

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description 'Pull finished FedRGBD run directories from the Jetson testbed (read-only on the node). Removes cleanly with scripts\uninstall_fetch_task.ps1.' | Out-Null

Write-Host ("registered '{0}': every {1} min, first run {2}" -f $TaskName, $IntervalMinutes, (Get-Date).AddMinutes(2).ToString('HH:mm'))
Write-Host ''
Write-Host 'Verify with:'
Write-Host ("  Get-ScheduledTaskInfo -TaskName {0}" -f $TaskName)
Write-Host ("  Start-ScheduledTask   -TaskName {0}    # run one pass now" -f $TaskName)
Write-Host ("  Get-Content logs\fetch.log -Tail 20 -Wait")
Write-Host ''
Write-Host 'Remove with:'
Write-Host ('  powershell -ExecutionPolicy Bypass -File scripts\uninstall_fetch_task.ps1')
