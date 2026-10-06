param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,

    [Parameter(Mandatory = $true)]
    [string]$StateDir,

    [ValidateSet("finnhub", "alpaca")]
    [string]$Source = "alpaca",

    [switch]$EnableHaikuShadow
)

$ErrorActionPreference = "Stop"
$resolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
$stateItem = Get-Item -LiteralPath $StateDir -ErrorAction Stop
if (-not $stateItem.PSIsContainer) {
    throw "StateDir must be an existing directory"
}
$resolvedState = $stateItem.FullName
$toolsRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$toolsRoot;$env:PYTHONPATH" } else { $toolsRoot }

if ($EnableHaikuShadow) {
    $env:SCANNER_JUDGE_SHADOW = "1"
    $env:SCANNER_SHADOW_LOG_PATH = Join-Path $resolvedState "judge_shadow.jsonl"
} else {
    $env:SCANNER_JUDGE_SHADOW = "0"
    Remove-Item Env:SCANNER_SHADOW_LOG_PATH -ErrorAction SilentlyContinue
}

& $resolvedPython -m momentum_scanner.scanner_movers_daemon `
    --state-dir $resolvedState `
    --source $Source
exit $LASTEXITCODE
