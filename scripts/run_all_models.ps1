<#
.SYNOPSIS
    Run the best conversion styles over a folder of images.

.DESCRIPTION
    Batch-converts every image in -InputDir once per style, writing each
    style's results into its own subdirectory under -OutputDir (avoids
    filename collisions between styles converting the same input file,
    and keeps results comparable side by side).

    By default, only non-experimental, recommended styles run (mirrors
    scripts/compare.py's RECOMMENDED_STYLES: excludes adaptive, gated,
    xdog -- not coloring-page candidates by their own measurement -- and
    chained, superseded as the CLI default by lineart-raster, see
    README's "Conversion styles" section). Pass -IncludeExperimental to
    run every registered style instead, or -Styles to pick an explicit
    list.

    A style requested but not currently registered (its optional extra
    isn't installed -- see README's "Extending with an ML engine") is
    reported and skipped up front, instead of failing silently partway
    through or drowning in one per-image error per file. A style that IS
    registered but is missing its checkpoint still fails per-image
    (surfaced by coloring-page itself as "Skipping <file>:
    WeightsMissingError ..."), since that's a per-file condition, not a
    per-style one (an explicit --weights-path could point at a file that
    only exists for some inputs' run, in principle).

    After conversion, runs scripts/report_quality.py over the same
    -InputDir for every style that actually ran, printing one PASS/FAIL
    table (ink coverage, enclosed/leaking regions, dangling endpoints)
    instead of just the files that were written.

.PARAMETER InputDir
    Directory of .jpg/.jpeg/.png photos to convert.

.PARAMETER OutputDir
    Directory to write one subdirectory per style into.

.PARAMETER Styles
    Optional list of style names to run instead of the recommended set,
    e.g. -Styles canny,chained.

.PARAMETER IncludeExperimental
    Run every registered style (from coloring_page.engines.registry.ENGINES),
    not just the recommended set. Ignored if -Styles is given.

.PARAMETER Detail
    toddler/child/adult, passed through as --detail (default: child).

.PARAMETER Format
    svg/pdf/png, passed through as --format (default: png).

.EXAMPLE
    .\scripts\run_all_models.ps1 -InputDir .\docs -OutputDir .\docs\_best_styles

.EXAMPLE
    .\scripts\run_all_models.ps1 -InputDir .\photos -OutputDir .\out -IncludeExperimental

.EXAMPLE
    .\scripts\run_all_models.ps1 -InputDir .\photos -OutputDir .\out -Styles lineart-raster,chained -Detail adult -Format svg
#>
param(
    [Parameter(Mandatory = $true)][string]$InputDir,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string[]]$Styles,
    [switch]$IncludeExperimental,
    [ValidateSet("toddler", "child", "adult")][string]$Detail = "child",
    [ValidateSet("svg", "pdf", "png")][string]$Format = "png"
)

$ErrorActionPreference = "Stop"

# Kept in sync by hand with scripts/compare.py's RECOMMENDED_STYLES exclusion
# set -- not queried dynamically from there, since that script isn't on
# Python's import path from a plain `python -c` invocation and isn't worth
# a PYTHONPATH dance for one shared constant.
$RecommendedExclusions = @("adaptive", "gated", "xdog", "chained")

$registryOutput = python -c "from coloring_page.engines.registry import ENGINES; print(' '.join(sorted(ENGINES)))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to query registered styles -- is the project installed (pip install -e .)?"
}
$RegisteredStyles = $registryOutput -split " "

if (-not $Styles) {
    if ($IncludeExperimental) {
        $Styles = $RegisteredStyles
    }
    else {
        $Styles = $RegisteredStyles | Where-Object { $RecommendedExclusions -notcontains $_ }
    }
}

$StylesToRun = @()
foreach ($Style in $Styles) {
    if ($RegisteredStyles -notcontains $Style) {
        Write-Warning "SKIPPED: '$Style' is not registered -- its optional extra likely isn't installed (see README's 'Extending with an ML engine' section)."
        continue
    }
    $StylesToRun += $Style
}

if ($StylesToRun.Count -eq 0) {
    throw "No requested style is currently registered; nothing to run."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

foreach ($Style in $StylesToRun) {
    Write-Host "== $Style ==" -ForegroundColor Cyan
    $styleOutputDir = Join-Path $OutputDir $Style
    python -m coloring_page.cli $InputDir $styleOutputDir --style $Style --detail $Detail --format $Format
}

Write-Host "`nDone converting. Outputs under $OutputDir, one subdirectory per style." -ForegroundColor Green

Write-Host "`n== Quality report (validate() over $InputDir) ==" -ForegroundColor Cyan
$reportArgs = @($InputDir)
foreach ($Style in $StylesToRun) {
    $reportArgs += "--style"
    $reportArgs += $Style
}
python scripts/report_quality.py @reportArgs
