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
    JSON, carry model_selection.selected_round AND the "power" record that
    scripts/run_matrix.py adds after its own checks, and not have been modified for
    $MinAgeMinutes minutes (default 15, longer than run_matrix's 10-minute finish
    grace). A run whose identity gate failed has no results.json (it is renamed) and
    is never fetched. A run still being written fails these tests and is skipped
    until the next pass.

    Runs are looked for in results/rev_* (the heterogeneous-power matrix), in
    every power-configuration namespace results/pc_<name>/rev_*, and in the camera
    namespace results/camera/<config>/{rev,diag}_camera_* (camera runs live only
    there, separate from FLAME); the local copy goes to the same relative path.

    Every pass runs scripts/block_report.py, which prints "COMPLETE <block> <n>" for
    every complete block. A pop-up is shown for each (block, n) not yet in
    logs/fetch_notified.state.json, and only then is it recorded there: delivery is
    at least once, and a pass that fails to show it retries on the next. The first
    pass after this was introduced records the blocks that were already complete,
    silently. A "milestone" in the config can attach its own message and commands to
    one block and run count; it fires only if the node's run_matrix.log shows the
    block ended cleanly (require_log_regex), and the commands are written to
    fetch.log. If the log cannot be read, nothing is recorded and the next pass
    tries again.

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

$MinAgeMinutes = if ($cfg.min_age_minutes) { [int]$cfg.min_age_minutes } else { 15 }
# results/rev_<run>, results/pc_<config>/rev_<run>, or
# results/camera/<config>/rev_camera_<run> | diag_camera_<run>
$RunPathRegex = '^results/((?:pc_[A-Za-z0-9_-]+/)?rev_[^/]+|camera/[A-Za-z0-9_-]+/(?:rev|diag)_camera_[^/]+)/results\.json$'

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
    # Windows PowerShell 5.1 turns ANY stderr output of a native command into a
    # terminating error under $ErrorActionPreference = 'Stop', even when redirected.
    # ssh's stderr is data here (logged below), never a reason to abort the pass.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out = & $SshExe @SshOpts $Remote $Command 2>$errFile
        $script:LastExit = $LASTEXITCODE
        $err = (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
        if ($script:LastExit -ne 0 -and $err) { Write-Log 'WARN' ("ssh stderr: {0}" -f $err.Trim()) }
        return $out
    } finally {
        $ErrorActionPreference = $prevEap
        Remove-Item $errFile -ErrorAction SilentlyContinue
    }
}

function Test-Reachable {
    $null = Invoke-Remote 'true'
    return ($script:LastExit -eq 0)
}

# --------------------------------------------------------------------------- #
# alerting on the node's own logs
# --------------------------------------------------------------------------- #
$AlertStateFile = Join-Path $LogDir 'fetch_alerts.state.json'
$AlertStateCap = 800          # keep the state file bounded; far more than a block emits

function Get-AlertPatterns {
    if ($cfg.alert_patterns) { return $cfg.alert_patterns }
    return @(
        [pscustomobject]@{ pattern = 'STOPPING';            ignore_case = $false },
        [pscustomobject]@{ pattern = 'DURDU';               ignore_case = $false },
        [pscustomobject]@{ pattern = 'BASLAMADI';           ignore_case = $false },
        [pscustomobject]@{ pattern = 'pre-?flight\s+FAIL';  ignore_case = $true  },
        # every decision of scripts/resume_after_reboot.py (logs/resume.log)
        [pscustomobject]@{ pattern = 'RESUME \[';           ignore_case = $false }
    )
}

function Get-RegexOptions {
    <#  Case folding here MUST be culture-invariant.

        This machine runs under tr-TR, where 'I' and 'i' are distinct letters: the
        capital of 'i' is 'İ' and the lowercase of 'I' is 'ı'. .NET's culture-aware
        case-insensitive matching therefore refuses to match 'fail' against 'FAIL',
        and PowerShell's -imatch inherits that. Verified on this machine:

            [regex]::IsMatch('fail','FAIL', IgnoreCase)                  -> False
            [regex]::IsMatch('fail','FAIL', IgnoreCase|CultureInvariant) -> True

        Every marker we watch for contains an i or an I -- STOPPING, BASLAMADI,
        "pre-flight FAIL" -- so without CultureInvariant the case-insensitive
        patterns would silently never fire, and the job would look healthy while
        missing exactly the events it exists to catch.
    #>
    param([bool]$IgnoreCase)
    $opts = [System.Text.RegularExpressions.RegexOptions]::CultureInvariant
    if ($IgnoreCase) { $opts = $opts -bor [System.Text.RegularExpressions.RegexOptions]::IgnoreCase }
    return $opts
}

function Show-Alert {
    <#  Never blocks. A scheduled pass must not sit waiting for someone to click OK,
        so msg.exe (which returns immediately) is tried first and the MessageBox
        fallback is launched as a detached process. #>
    param([string]$Body)
    $msgExe = Join-Path $env:SystemRoot 'System32\msg.exe'
    if (Test-Path $msgExe) {
        # msg.exe rejects a message longer than ~255 characters ("Invalid parameter(s)"),
        # and under $ErrorActionPreference = 'Stop' its stderr used to abort the whole
        # alert scan -- no pop-up and no state saved (2026-09-28). Shorten, never throw.
        $text = "FedRGBD testbed: " + $Body
        $tail = ' ... (full text: logs\fetch.log)'
        if ($text.Length -gt 250) { $text = $text.Substring(0, 250 - $tail.Length) + $tail }
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $msgExe * /TIME:3600 $text 2>$null | Out-Null
            $msgExit = $LASTEXITCODE
        } catch {
            $msgExit = -1
        } finally { $ErrorActionPreference = $prevEap }
        if ($msgExit -eq 0) { return }
        Write-Log 'WARN' ("msg.exe failed (exit {0}); falling back to a message box" -f $msgExit)
    }
    # msg.exe is absent on Home editions; fall back to a detached message box.
    $safe = $Body -replace "'", "''"
    $inner = "Add-Type -AssemblyName System.Windows.Forms; " +
             "[void][System.Windows.Forms.MessageBox]::Show('$safe','FedRGBD testbed alert'," +
             "[System.Windows.Forms.MessageBoxButtons]::OK,[System.Windows.Forms.MessageBoxIcon]::Warning)"
    try {
        Start-Process -FilePath 'powershell.exe' `
            -ArgumentList '-NoProfile', '-WindowStyle', 'Hidden', '-Command', $inner `
            -WindowStyle Hidden | Out-Null
    } catch {
        Write-Log 'WARN' ("could not raise a desktop alert: {0}" -f $_.Exception.Message)
    }
}

function Invoke-AlertScan {
    <#  Read-only scan of the node's logs for failure markers. Each distinct matching
        line alerts exactly once: the line text is fingerprinted and remembered in
        logs/fetch_alerts.state.json, so an hourly pass over the same log is silent.
        run_matrix.log lines carry timestamps, so a genuinely repeated event is a
        different line and does alert again. #>
    $patterns = Get-AlertPatterns
    $tailLines = if ($cfg.alert_log_tail_lines) { [int]$cfg.alert_log_tail_lines } else { 400 }

    $seen = @{}
    if (Test-Path $AlertStateFile) {
        try {
            foreach ($h in (Get-Content $AlertStateFile -Raw | ConvertFrom-Json).seen) { $seen[$h] = $true }
        } catch { Write-Log 'WARN' 'alert state file unreadable; treating every match as new' }
    }
    $firstRun = ($seen.Count -eq 0) -and (-not (Test-Path $AlertStateFile))

    $sources = @(
        @{ name = 'chain.log';      path = '~/chain.log' },
        @{ name = 'run_matrix.log'; path = ("'{0}/logs/run_matrix.log'" -f $RemoteRepo) },
        @{ name = 'resume.log';     path = ("'{0}/logs/resume.log'" -f $RemoteRepo) }
    )

    $new = @()
    $order = @()
    foreach ($src in $sources) {
        $text = Invoke-Remote ("tail -n {0} {1} 2>/dev/null" -f $tailLines, $src.path)
        if ($script:LastExit -ne 0 -or -not $text) { continue }
        foreach ($line in ($text -split "`n")) {
            $line = $line.TrimEnd()
            if (-not $line.Trim()) { continue }
            $hit = $false
            foreach ($p in $patterns) {
                $ic = $false
                if ($null -ne $p.ignore_case) { $ic = [bool]$p.ignore_case }
                if ([regex]::IsMatch($line, $p.pattern, (Get-RegexOptions $ic))) { $hit = $true; break }
            }
            if (-not $hit) { continue }
            $key = '{0}|{1}' -f $src.name, $line
            $sha = [System.BitConverter]::ToString(
                [System.Security.Cryptography.SHA256]::Create().ComputeHash(
                    [System.Text.Encoding]::UTF8.GetBytes($key))).Replace('-', '').Substring(0, 32)
            $order += $sha
            if (-not $seen.ContainsKey($sha)) {
                $seen[$sha] = $true
                $new += [pscustomobject]@{ source = $src.name; line = $line; hash = $sha }
            }
        }
    }

    if ($new.Count -gt 0) {
        if ($firstRun) {
            # Do not fire a pop-up for history that predates alerting being switched on;
            # record it as seen and log it quietly instead.
            foreach ($n in $new) { Write-Log 'INFO' ("pre-existing marker in {0}: {1}" -f $n.source, $n.line) }
            Write-Log 'INFO' ("alert baseline established from {0} existing marker line(s); future ones will alert" -f $new.Count)
        } else {
            foreach ($n in $new) { Write-Log 'ALERT' ("{0}: {1}" -f $n.source, $n.line) }
            $head = ($new | Select-Object -First 3 | ForEach-Object { "[$($_.source)] $($_.line)" }) -join "`n"
            if ($new.Count -gt 3) { $head += ("`n... and {0} more (see logs\fetch.log)" -f ($new.Count - 3)) }
            Show-Alert $head
        }
    }

    # Remember exactly the markers still visible in the tail window, bounded. A marker
    # that has scrolled out cannot match again, so dropping it is safe and keeps the
    # state file from growing for the whole 6-9 day run.
    $keep = @($order | Select-Object -Unique | Select-Object -Last $AlertStateCap)
    $state = [pscustomobject]@{ updated = (Get-Date -Format 'o'); seen = $keep }
    try { $state | ConvertTo-Json -Depth 3 | Set-Content -Path $AlertStateFile -Encoding utf8 }
    catch { Write-Log 'WARN' 'could not write the alert state file; markers may alert again next pass' }

    return $new.Count
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

# One call for every candidate's checksum, restricted to results.json files that have
# not changed for $MinAgeMinutes minutes. `md5sum` on a file being written still
# returns something, so the JSON validity check below is what actually gates a fetch.
$remoteList = Invoke-Remote ("cd '{0}' && find results -maxdepth 4 -name results.json -mmin +{1} -regextype posix-extended -regex 'results/((pc_[A-Za-z0-9_-]+/)?rev_[^/]+|camera/[A-Za-z0-9_-]+/(rev|diag)_camera_[^/]+)/results[.]json' -exec md5sum {{}} + 2>/dev/null" -f $RemoteRepo, $MinAgeMinutes)
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
        if ($p -match $RunPathRegex) { $tracked[$Matches[1]] = $true }
    }
    Write-Log 'INFO' ("{0} run(s) already committed; their local copies are authoritative and will not be compared" -f $tracked.Count)
} catch {
    Write-Log 'WARN' 'could not list git-tracked runs; every differing local copy will be reported'
}

$fetched = 0; $skipped = 0; $pending = 0; $warned = 0; $committed = 0
foreach ($line in ($remoteList -split "`n")) {
    $line = $line.Trim()
    if (-not $line) { continue }
    if ($line -notmatch '^([0-9a-f]{32})\s+(\S+)$') { continue }
    $remoteMd5 = $Matches[1]
    if ($Matches[2] -notmatch $RunPathRegex) { continue }
    $run = $Matches[1]                      # rev_<run> or pc_<config>/rev_<run>

    $localRun = Join-Path $LocalResults ($run -replace '/', '\')
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
    if (-not ($parsed.PSObject.Properties.Name -contains 'power')) {
        # run_matrix adds "power" only after its own checks (identity gates, power modes)
        Write-Log 'INFO' ("{0}: no power record yet -- the runner has not finished with it, skipping" -f $run)
        $pending++; continue
    }

    if ($DryRun) {
        Write-Log 'INFO' ("{0}: WOULD fetch (selected_round={1})" -f $run, $selected)
        $fetched++; continue
    }

    # Copy into a staging name first, so an interrupted scp can never leave a
    # half-written directory that a later pass would mistake for a finished run.
    $staging = Join-Path $LocalResults ('.incoming_' + ($run -replace '/', '__'))
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
    $parent = Split-Path -Parent $localRun
    if (-not (Test-Path $parent)) { New-Item -ItemType Directory -Path $parent | Out-Null }
    Move-Item $staging $localRun
    $npz = (Get-ChildItem -Path (Join-Path $localRun 'predictions') -Filter *.npz -ErrorAction SilentlyContinue).Count
    Write-Log 'INFO' ("{0}: fetched (selected_round={1}, {2} prediction files)" -f $run, $selected, $npz)
    $fetched++
}

Write-Log 'INFO' ("fetch done: {0} fetched, {1} already present, {2} in progress, {3} committed-and-differing (ignored), {4} warnings" -f $fetched, $skipped, $pending, $committed, $warned)

# --------------------------------------------------------------------------- #
# block completion -> mechanical pipeline into a scratch directory
# --------------------------------------------------------------------------- #
$NotifiedStateFile = Join-Path $LogDir 'fetch_notified.state.json'

function Invoke-BlockNotifications {
    <#  One pop-up per complete (block, run count) not yet notified, recorded only
        AFTER it was shown: a pass that fails half-way delivers it on the next pass.
        A configured milestone for that block and run count replaces the generic text
        with its own message and writes its commands to fetch.log -- but only if the
        node's run_matrix.log shows the block ended cleanly; otherwise the pop-up
        says it did NOT end cleanly. If the log cannot be read, nothing is recorded. #>
    param([string[]]$ReportLines)
    $notified = @{}
    $baseline = -not (Test-Path $NotifiedStateFile)
    if (-not $baseline) {
        try { foreach ($k in (Get-Content $NotifiedStateFile -Raw | ConvertFrom-Json).notified) { $notified[$k] = $true } }
        catch { Write-Log 'WARN' 'notification state unreadable; not notifying this pass'; return }
    }
    $changed = $false
    foreach ($l in $ReportLines) {
        if ($l -notmatch '^COMPLETE\s+(\S+)\s+(\d+)\s*$') { continue }
        $block = $Matches[1]; $n = [int]$Matches[2]
        $key = '{0}|{1}' -f $block, $n
        if ($notified.ContainsKey($key)) { continue }
        if ($baseline) {
            # blocks complete before notifications existed: record, do not pop up
            Write-Log 'INFO' ("notification baseline: block {0} already complete ({1} runs)" -f $block, $n)
            $notified[$key] = $true; $changed = $true
            continue
        }
        try {
            $milestone = $null
            foreach ($m in @($cfg.milestones)) {
                if ($m -and $m.block -eq $block -and [int]$m.runs -eq $n) { $milestone = $m; break }
            }
            if (-not $milestone) {
                Write-Log 'NOTIFY' ("block {0} complete ({1} runs)" -f $block, $n)
                Show-Alert ("block {0} complete ({1} runs) -- see logs\fetch.log" -f $block, $n)
            } else {
                $clean = $true
                if ($milestone.require_log_regex) {
                    $tailN = if ($cfg.alert_log_tail_lines) { [int]$cfg.alert_log_tail_lines } else { 400 }
                    $text = Invoke-Remote ("tail -n {0} '{1}/logs/run_matrix.log'" -f $tailN, $RemoteRepo)
                    if ($script:LastExit -ne 0 -or -not $text) {
                        Write-Log 'WARN' ("{0}: could not read run_matrix.log to confirm a clean end; will retry next pass" -f $key)
                        continue
                    }
                    $clean = [regex]::IsMatch(($text -join "`n"), $milestone.require_log_regex, (Get-RegexOptions $false))
                }
                if ($clean) {
                    Write-Log 'MILESTONE' $milestone.message
                    foreach ($c in @($milestone.commands)) { if ($c) { Write-Log 'MILESTONE' ("    " + $c) } }
                    Show-Alert ($milestone.message + " -- the exact commands are in logs\fetch.log")
                } else {
                    $msg = ("block {0} complete ({1} runs) but run_matrix.log does NOT show a clean end ({2}); do not switch anything, check the node" -f $block, $n, $milestone.require_log_regex)
                    Write-Log 'ALERT' $msg
                    Show-Alert $msg
                }
            }
            $notified[$key] = $true; $changed = $true
        } catch {
            Write-Log 'WARN' ("notification for {0} failed ({1}); will retry next pass" -f $key, $_.Exception.Message)
        }
    }
    if ($changed -or $baseline) {
        $state = [pscustomobject]@{ updated = (Get-Date -Format 'o'); notified = @($notified.Keys | Sort-Object) }
        try { $state | ConvertTo-Json -Depth 3 | Set-Content -Path $NotifiedStateFile -Encoding utf8 }
        catch { Write-Log 'WARN' 'could not write the notification state; a pop-up may repeat' }
    }
}

# Every pass, not only passes that fetched something: block_report is idempotent, and
# a notification that failed on the pass that completed a block must be retried even
# though the node then sits idle (after 5a it waits for a human to switch modes).
if (-not $DryRun) {
    $python = $cfg.python
    $reporter = Join-Path $PSScriptRoot 'block_report.py'
    if ($python -and (Test-Path $python) -and (Test-Path $reporter)) {
        Write-Log 'INFO' 'checking for complete blocks'
        $prevEap = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'      # stderr of a native command must not throw
        try {
            $reportLines = @(& $python $reporter --repo $RepoRoot 2>&1 | ForEach-Object { "$_" })
            $reportExit = $LASTEXITCODE
        } finally { $ErrorActionPreference = $prevEap }
        foreach ($l in $reportLines) { if ($l) { Write-Log 'INFO' ("block_report: {0}" -f $l) } }
        if ($reportExit -ne 0) { Write-Log 'WARN' ("block_report exited {0}" -f $reportExit) }
        try { Invoke-BlockNotifications $reportLines }
        catch { Write-Log 'WARN' ("block notification failed: {0}" -f $_.Exception.Message) }
    } else {
        Write-Log 'WARN' 'python or block_report.py not configured; skipping block reports'
    }
}

# A short tail of the node's own log, so one file here shows what the testbed is doing.
$tail = Invoke-Remote ("tail -n {0} '{1}/logs/run_matrix.log' 2>/dev/null" -f ([int]($cfg.log_tail_lines | ForEach-Object { if ($_) { $_ } else { 5 } })), $RemoteRepo)
if ($script:LastExit -eq 0 -and $tail) {
    foreach ($l in ($tail -split "`n")) { if ($l.Trim()) { Write-Log 'NODE' $l.Trim() } }
}

# Failure markers in the node's own logs. Read-only, and each distinct line alerts once.
try {
    $alerts = Invoke-AlertScan
    if ($alerts -gt 0) { Write-Log 'INFO' ("{0} new alert line(s) this pass" -f $alerts) }
} catch {
    # An alerting fault must never fail the fetch: the runs are the point.
    Write-Log 'WARN' ("alert scan failed: {0}" -f $_.Exception.Message)
}
exit 0
