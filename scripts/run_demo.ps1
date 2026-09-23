param([int]$Condition = 1, [int]$Seed = 42, [string]$Output = "runs/demo")
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
python -m electrowinning_rl demo --config "$ProjectRoot/configs/demo.json" --condition $Condition --seed $Seed --output $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
