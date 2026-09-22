<#
.SYNOPSIS
    Pull finished FedRGBD run directories from the Jetson testbed to this desktop.

.DESCRIPTION
    Runs on the Windows desktop only, never on a Jetson. It is strictly READ-ONLY on
    the remote node: the only remote commands it ever issues are

        ls / test        list results/ and check a path exists
        md5sum           checksum a remote results.json
        cat              read a remote results.json and the tail of run_matrix.log
        scp (pull)       copy a finished run directory here

    No git, no writes, no deletes, no process control, ever. Node A is running the
    block chain and a stray write there would cost days.

    A run is fetched only when it is FINISHED: its remote results.json must parse as
    JSON and carry model_selection.selected_round. A run still being written fails
    that test and is skipped until the next pass.

    Already-present runs are skipped by md5 of results.json. If a local copy exists
    and differs from the remote, nothing is overwritten -- the difference is logged
    as a warning and the run is left alone. This job doubles as the backup against
    SD-card failure on Node A, so it must never destroy the local copy.

    Windows OpenSSH is used explicitly (C:\Windows\System32\OpenSSH\ssh.exe), because
    the key lives in the Windows ssh-agent, which Git Bash's own ssh cannot see.
    Every remote call uses BatchMode=yes and a connect timeout, so the job fails fast
    and never hangs a scheduled task.

.PARAMETER ConfigPath
    JSON config; see scripts/fetch_results.config.example.json. Defaults to
    scripts/fetch_results.config.json (gitignored -- it names the host and user).

.PARAMETER DryRun
    Do everything except the scp: useful for the by-hand test before scheduling.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_results.ps1 -DryRun
#>
[CmdletBinding()]
param(
    [string]$ConfigPath,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $RepoRoot 'logs'
$LogFile = Join-Path $LogDir 'fetch.log'

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Write-Log {
    param([string]$Level, [string]$Message)
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Add-Content -Path $LogFile -Value $line -Encoding utf8
    Write-Host $line
}

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
if (-not $ConfigPath) { $ConfigPath = Join-Path $PSScriptRoot 'fetch_results.config.json' }
if (-not (Test-Path $ConfigPath)) {
    Write-Log 'ERROR' ("no config at {0} -- copy fetch_results.config.example.json and fill it in" -f $ConfigPath)
    exit 2
}
$cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json

$SshExe = 'C:\Windows\System32\OpenSSH\ssh.exe'
$ScpExe = 'C:\Windows\System32\OpenSSH\scp.exe'
foreach ($exe in @($SshExe, $ScpExe)) {
    if (-not (Test-Path $exe)) { Write-Log 'ERROR' "missing $exe (install the Windows OpenSSH client)"; exit 2 }
}

$Remote = '{0}@{1}' -f $cfg.user, $cfg.host
$RemoteRepo = $cfg.remote_repo.TrimEnd('/')
$LocalResults = Join-Path $RepoRoot 'results'
$ConnectTimeout = if ($cfg.connect_timeout_s) { [int]$cfg.connect_timeout_s } else { 15 }
$SshOpts = @(
    '-o', 'BatchMode=yes',                    # never prompt; fail instead
    '-o', ("ConnectTimeout={0}" -f $ConnectTimeout),
    '-o', 'StrictHostKeyChecking=accept-new'
)
if ($cfg.identity_file) { $SshOpts += @('-i', $cfg.identity_file) }

# --------------------------------------------------------------------------- #
# remote helpers -- the complete set of remote operations this script performs
# --------------------------------------------------------------------------- #
function Invoke-Remote {
    <#  Run one read-only command on the node. Returns stdout; $script:LastExit holds
        the exit code. stderr is captured into the log rather than the return value. #>
    param([string]$Command)
    $errFile = [System.IO.Path]::GetTempFileName()
    try {
        $out = & $SshExe @SshOpts $Remote $Command 2>$errFile
        $script:LastExit = $LASTEXITCODE
        $err = (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
        if ($script:LastExit -ne 0 -and $err) { Write-Log 'WARN' ("ssh stderr: {0}" -f $err.Trim()) }
        return $out
    } finally { Remove-Item $errFile -ErrorAction SilentlyContinue }
}

function Test-Reachable {
    $null = Invoke-Remote 'true'
    return ($script:LastExit -eq 0)
}

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
Write-Log 'INFO' ("fetch start -> {0}:{1}" -f $Remote, $RemoteRepo)

if (-not (Test-Reachable)) {
    # auth failure, node down, network gone: log and leave. Never retry in a loop,
    # never hang -- the task simply runs again in an hour.
    Write-Log 'ERROR' ("cannot reach {0} (ssh exit {1}); giving up this pass" -f $Remote, $script:LastExit)
    exit 1
}

# One call for every candidate's checksum. `md5sum` on a file being written still
# returns something, so the JSON validity check below is what actually gates a fetch.
$remoteList = Invoke-Remote ("cd '{0}' && md5sum results/rev_*/results.json 2>/dev/null" -f $RemoteRepo)
if ($script:LastExit -ne 0 -and -not $remoteList) {
    Write-Log 'INFO' 'no rev_* runs on the node yet'
    exit 0
}

# Runs already committed to git (the desktop-GPU baselines) legitimately differ from
# the node's own copies and are not this job's business. Without this, ~30 warnings
# would fire every hour and the one warning that matters -- a run we fetched changing
# underneath us -- would be lost in them.
$tracked = @{}
try {
    $gitOut = & git -C $RepoRoot ls-files 'results' 2>$null
    foreach ($p in ($gitOut -split "`n")) {
        if ($p -match '^results/(rev_[^/]+)/results\.json$') { $tracked[$Matches[1]] = $true }
    }
    Write-Log 'INFO' ("{0} run(s) already committed; their local copies are authoritative and will not be compared" -f $tracked.Count)
} catch {
    Write-Log 'WARN' 'could not list git-tracked runs; every differing local copy will be reported'
}

$fetched = 0; $skipped = 0; $pending = 0; $warned = 0; $committed = 0
foreach ($line in ($remoteList -split "`n")) {
    $line = $line.Trim()
    if (-not $line) { continue }
    if ($line -notmatch '^([0-9a-f]{32})\s+results/(rev_[^/]+)/results\.json$') { continue }
    $remoteMd5 = $Matches[1]
    $run = $Matches[2]

    $localRun = Join-Path $LocalResults $run
    $localJson = Join-Path $localRun 'results.json'

    if (Test-Path $localJson) {
        $localMd5 = (Get-FileHash -Path $localJson -Algorithm MD5).Hash.ToLower()
        if ($localMd5 -eq $remoteMd5) { $skipped++; continue }
        if ($tracked.ContainsKey($run)) {
            # Committed reference data (e.g. the desktop-GPU baselines). The local copy
            # is the published one; the node's differing copy is expected and irrelevant.
            $committed++
            continue
        }
        # Present, untracked and different: a run we fetched has changed on the node.
        # Never overwrite -- this local copy is also the backup against SD-card failure.
        Write-Log 'WARN' ("{0}: local results.json differs from remote (local {1}, remote {2}); NOT overwriting" -f $run, $localMd5.Substring(0,8), $remoteMd5.Substring(0,8))
        $warned++
        continue
    }

    # Finished? The remote results.json must parse and carry a selected round.
    $json = Invoke-Remote ("cat '{0}/results/{1}/results.json' 2>/dev/null" -f $RemoteRepo, $run)
    if ($script:LastExit -ne 0 -or -not $json) { $pending++; continue }
    try { $parsed = $json | ConvertFrom-Json } catch {
        Write-Log 'INFO' ("{0}: results.json does not parse yet -- run in progress, skipping" -f $run)
        $pending++; continue
    }
    $selected = $null
    if ($parsed.PSObject.Properties.Name -contains 'model_selection' -and $parsed.model_selection) {
        $selected = $parsed.model_selection.selected_round
    }
    if ($null -eq $selected) {
        Write-Log 'INFO' ("{0}: no model_selection.selected_round -- run in progress, skipping" -f $run)
        $pending++; continue
    }

    if ($DryRun) {
        Write-Log 'INFO' ("{0}: WOULD fetch (selected_round={1})" -f $run, $selected)
        $fetched++; continue
    }

    # Copy into a staging name first, so an interrupted scp can never leave a
    # half-written directory that a later pass would mistake for a finished run.
    $staging = Join-Path $LocalResults ('.incoming_' + $run)
    if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
    & $ScpExe @SshOpts '-r' ("{0}:{1}/results/{2}" -f $Remote, $RemoteRepo, $run) $staging 2>&1 |
        ForEach-Object { if ($_) { Write-Log 'DEBUG' $_ } }
    if ($LASTEXITCODE -ne 0) {
        Write-Log 'ERROR' ("{0}: scp failed (exit {1})" -f $run, $LASTEXITCODE)
        if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
        continue
    }
    $stagedJson = Join-Path $staging 'results.json'
    if (-not (Test-Path $stagedJson)) {
        Write-Log 'ERROR' ("{0}: fetched copy has no results.json; discarding" -f $run)
        Remove-Item $staging -Recurse -Force
        continue
    }
    $gotMd5 = (Get-FileHash -Path $stagedJson -Algorithm MD5).Hash.ToLower()
    if ($gotMd5 -ne $remoteMd5) {
        Write-Log 'WARN' ("{0}: checksum changed during transfer (the run may have been rewritten); discarding, will retry next pass" -f $run)
        Remove-Item $staging -Recurse -Force
        continue
    }
    Move-Item $staging $localRun
    $npz = (Get-ChildItem -Path (Join-Path $localRun 'predictions') -Filter *.npz -ErrorAction SilentlyContinue).Count
    Write-Log 'INFO' ("{0}: fetched (selected_round={1}, {2} prediction files)" -f $run, $selected, $npz)
    $fetched++
}

Write-Log 'INFO' ("fetch done: {0} fetched, {1} already present, {2} in progress, {3} committed-and-differing (ignored), {4} warnings" -f $fetched, $skipped, $pending, $committed, $warned)

# --------------------------------------------------------------------------- #
# block completion -> mechanical pipeline into a scratch directory
# --------------------------------------------------------------------------- #
if ($fetched -gt 0 -and -not $DryRun) {
    $python = $cfg.python
    $reporter = Join-Path $PSScriptRoot 'block_report.py'
    if ($python -and (Test-Path $python) -and (Test-Path $reporter)) {
        Write-Log 'INFO' 'checking for newly complete blocks'
        & $python $reporter --repo $RepoRoot 2>&1 | ForEach-Object { if ($_) { Write-Log 'INFO' ("block_report: {0}" -f $_) } }
        if ($LASTEXITCODE -ne 0) { Write-Log 'WARN' ("block_report exited {0}" -f $LASTEXITCODE) }
    } else {
        Write-Log 'WARN' 'python or block_report.py not configured; skipping block reports'
    }
}

# A short tail of the node's own log, so one file here shows what the testbed is doing.
$tail = Invoke-Remote ("tail -n {0} '{1}/logs/run_matrix.log' 2>/dev/null" -f ([int]($cfg.log_tail_lines | ForEach-Object { if ($_) { $_ } else { 5 } })), $RemoteRepo)
if ($script:LastExit -eq 0 -and $tail) {
    foreach ($l in ($tail -split "`n")) { if ($l.Trim()) { Write-Log 'NODE' $l.Trim() } }
}
exit 0
