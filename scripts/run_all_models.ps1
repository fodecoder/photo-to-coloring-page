<#
.SYNOPSIS
    Run every registered conversion style over a folder of images.

.DESCRIPTION
    Batch-converts every image in -InputDir once per registered style
    (queried live from coloring_page.engines.registry.ENGINES, so this
    stays in sync automatically as styles are added/removed), writing
    each style's results into its own subdirectory under -OutputDir.

    Styles that need pretrained weights that haven't been downloaded yet
    (informative_drawings, anime2sketch -- see README's "Extending with
    an ML engine" section) are skipped per-image by the CLI itself, with
    a warning; this script does not stop early because one style failed.

.PARAMETER InputDir
    Directory of .jpg/.jpeg/.png photos to convert.

.PARAMETER OutputDir
    Directory to write one subdirectory per style into.

.PARAMETER Styles
    Optional list of style names to run instead of every registered
    style, e.g. -Styles canny,chained.

.EXAMPLE
    .\scripts\run_all_models.ps1 -InputDir .\docs -OutputDir .\docs\_all_styles

.EXAMPLE
    .\scripts\run_all_models.ps1 -InputDir .\photos -OutputDir .\out -Styles canny,xdog
#>
param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string[]]$Styles
)

$ErrorActionPreference = "Stop"

if (-not $Styles) {
    $registryOutput = python -c "from coloring_page.engines.registry import ENGINES; print(' '.join(sorted(ENGINES)))"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query registered styles -- is the project installed (pip install -e .)?"
    }
    $Styles = $registryOutput -split " "
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($Style in $Styles) {
    Write-Host "== $Style ==" -ForegroundColor Cyan
    $styleOutputDir = Join-Path $OutputDir $Style
    python -m coloring_page.cli $InputDir $styleOutputDir --style $Style
}

Write-Host "Done. Outputs under $OutputDir, one subdirectory per style." -ForegroundColor Green
