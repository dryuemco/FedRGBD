<#
.SYNOPSIS
    Remove the hourly FedRGBD result-fetch task from this desktop.

.DESCRIPTION
    Unregisters the scheduled task. It does not touch results/, logs/fetch.log, the
    config file or anything on the Jetsons -- only the task registration goes away, so
    running this is always safe and never loses data.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\uninstall_fetch_task.ps1
#>
[CmdletBinding()]
param([string]$TaskName = 'FedRGBD-FetchResults')

$ErrorActionPreference = 'Stop'

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host ("no scheduled task named '{0}' -- nothing to remove" -f $TaskName)
    exit 0
}

if ($task.State -eq 'Running') {
    Write-Host 'a pass is running; stopping it first'
    Stop-ScheduledTask -TaskName $TaskName
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host ("removed '{0}'" -f $TaskName)
Write-Host 'Kept, as they are data rather than configuration: results\, logs\fetch.log, scripts\fetch_results.config.json, scratch\block_reports\.'
