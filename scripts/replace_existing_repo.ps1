param(
    [Parameter(Mandatory=$true)]
    [string]$ExistingRepoPath
)

$ErrorActionPreference = "Stop"
$source = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$target = (Resolve-Path $ExistingRepoPath).Path

if (-not (Test-Path (Join-Path $target ".git"))) {
    throw "The target is not a Git repository: $target"
}

Write-Host "Replacing project files in: $target"

Get-ChildItem -Path $target -Force |
    Where-Object { $_.Name -ne ".git" } |
    Remove-Item -Recurse -Force

Get-ChildItem -Path $source -Force |
    Where-Object { $_.Name -ne ".git" } |
    Copy-Item -Destination $target -Recurse -Force

Set-Location $target
git add -A
git status

Write-Host ""
Write-Host "Review the status above. Then run:"
Write-Host 'git commit -m "Refactor Colab notebook into server-ready Python scripts"'
Write-Host "git push origin main"
