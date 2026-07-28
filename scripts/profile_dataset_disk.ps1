<#
.SYNOPSIS
    Mede, no sistema de arquivos nativo do Windows, o espaço ocupado pelos datasets locais.

.DESCRIPTION
    A leitura dos arrays e rótulos continua no Python/WSL, pois usa os adaptadores
    do benchmark. Esta etapa existe porque uma varredura de centenas de milhares
    de metadados por /mnt/c pode ser muito mais lenta no WSL. O resultado é um
    JSON consumido por scripts/profile_datasets.py --disk-profile.
#>

[CmdletBinding()]
param(
    [string]$DatasetRoot,
    [string]$Output,
    [string[]]$DatasetNames = @(
        'mnist', 'fashion_mnist', 'kmnist', 'emnist_balanced', 'cifar10',
        'cifar100_coarse', 'cinic10', 'svhn', 'gtsrb', 'fer2013'
    )
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$scriptDirectory = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) { $DatasetRoot = Join-Path $scriptDirectory '..\datasets' }
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $scriptDirectory '..\outputs\dataset-inventory\disk_profile.json' }

function Format-ByteSize {
    param([Int64]$Bytes)
    $units = @('B', 'KiB', 'MiB', 'GiB', 'TiB')
    [double]$value = $Bytes
    foreach ($unit in $units) {
        if ($value -lt 1024 -or $unit -eq $units[-1]) {
            return ('{0:N2} {1}' -f $value, $unit)
        }
        $value /= 1024
    }
}

$resolvedRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$profiles = [ordered]@{}

foreach ($name in $DatasetNames) {
    $root = Join-Path $resolvedRoot $name
    if (-not (Test-Path -LiteralPath $root -PathType Container)) {
        throw "Dataset não encontrado: $root"
    }
    Write-Host "[disk] $name"
    [Int64]$totalBytes = 0
    [Int64]$fileCount = 0
    $extensions = @{}
    Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object {
        [Int64]$totalBytes += $_.Length
        [Int64]$fileCount += 1
        $extension = if ([string]::IsNullOrWhiteSpace($_.Extension)) { '[sem extensao]' } else { $_.Extension.ToLowerInvariant() }
        if (-not $extensions.ContainsKey($extension)) {
            $extensions[$extension] = [ordered]@{ extension = $extension; files = [Int64]0; bytes = [Int64]0 }
        }
        [Int64]$extensions[$extension].files += 1
        [Int64]$extensions[$extension].bytes += $_.Length
    }
    $profiles[$name] = [ordered]@{
        root_bytes = $totalBytes
        root_bytes_human = Format-ByteSize $totalBytes
        root_file_count = $fileCount
        extension_distribution = @(
            $extensions.Values |
                Sort-Object -Property @{ Expression = 'bytes'; Descending = $true }, extension |
                ForEach-Object { [ordered]@{ extension = $_.extension; files = $_.files; bytes = $_.bytes } }
        )
    }
}

$payload = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    measurement_environment = 'Windows native filesystem enumeration'
    datasets = $profiles
}
$outputPath = [System.IO.Path]::GetFullPath($Output)
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath)
$json = $payload | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText($outputPath, $json + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
Write-Host "Perfil de disco salvo em: $outputPath"
