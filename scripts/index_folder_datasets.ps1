<#
.SYNOPSIS
    Indexa CINIC-10 e GTSRB no Windows para a etapa de inventário.

.DESCRIPTION
    Os dois datasets são formados por muitos arquivos. Em WSL, listar todos os
    caminhos de um volume montado em /mnt/c é substancialmente mais lento. Este
    script lê os mesmos diretórios e CSVs no Windows, produzindo contagens
    exatas por classe/split e uma amostra de caminhos para a inspeção de
    formatos feita em Python. Não move, edita ou baixa dados.
#>

[CmdletBinding()]
param(
    [string]$DatasetRoot,
    [string]$Output,
    [int]$SamplePerClass = 24
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($SamplePerClass -lt 1) { throw 'SamplePerClass deve ser positivo.' }
$scriptDirectory = Split-Path -Parent $PSCommandPath
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) { $DatasetRoot = Join-Path $scriptDirectory '..\datasets' }
if ([string]::IsNullOrWhiteSpace($Output)) { $Output = Join-Path $scriptDirectory '..\outputs\dataset-inventory\folder_index.json' }

function Convert-ToProjectRelativePath {
    param([string]$Path, [string]$ProjectRoot)
    $rootWithSeparator = $ProjectRoot.TrimEnd('\') + '\'
    if (-not $Path.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "O caminho '$Path' não está abaixo do projeto '$ProjectRoot'."
    }
    return $Path.Substring($rootWithSeparator.Length).Replace('\', '/')
}

function New-ClassRows {
    param([string[]]$ClassNames, [string[]]$Splits)
    $rows = @()
    for ($index = 0; $index -lt $ClassNames.Count; $index++) {
        $row = [ordered]@{ class_index = $index; class_name = [string]$ClassNames[$index]; total = [Int64]0 }
        foreach ($split in $Splits) { $row["split_$split"] = [Int64]0 }
        $rows += $row
    }
    return $rows
}

function Get-RowSplitTotal {
    param([object[]]$Rows, [string]$Split)
    [Int64]$total = 0
    foreach ($row in $Rows) { [Int64]$total += $row["split_$Split"] }
    return $total
}

function Get-Cinic10Index {
    param([string]$Root, [string]$ProjectRoot, [int]$SampleLimit)
    $splits = @('train', 'valid', 'test')
    $classNames = @(
        Get-ChildItem -LiteralPath (Join-Path $Root 'train') -Directory |
            Sort-Object -Property Name |
            Select-Object -ExpandProperty Name
    )
    if ($classNames.Count -eq 0) { throw "Classes CINIC-10 ausentes em $Root" }
    $rows = @(New-ClassRows -ClassNames $classNames -Splits $splits)
    $samples = New-Object 'System.Collections.Generic.List[string]'
    foreach ($split in $splits) {
        foreach ($index in 0..($classNames.Count - 1)) {
            $classDir = Join-Path (Join-Path $Root $split) $classNames[$index]
            if (-not (Test-Path -LiteralPath $classDir -PathType Container)) { continue }
            [Int64]$count = @(Get-ChildItem -LiteralPath $classDir -File -Force).Count
            [Int64]$rows[$index]["split_$split"] = $count
            [Int64]$rows[$index].total += $count
            if ($samples.Count -lt (($index + 1) * $SampleLimit)) {
                $needed = (($index + 1) * $SampleLimit) - $samples.Count
                Get-ChildItem -LiteralPath $classDir -File -Force |
                    Sort-Object -Property Name |
                    Select-Object -First $needed |
                    ForEach-Object { $samples.Add((Convert-ToProjectRelativePath -Path $_.FullName -ProjectRoot $ProjectRoot)) }
            }
        }
    }
    $splitTotals = [ordered]@{}
    foreach ($split in $splits) { $splitTotals[$split] = Get-RowSplitTotal -Rows $rows -Split $split }
    return [ordered]@{
        class_names = $classNames
        class_distribution = $rows
        original_splits = $splitTotals
        sample_paths = @($samples)
        image_inspection_scope = "amostra estratificada de ate $SampleLimit arquivo(s) por classe do split train; as contagens incluem train/valid/test"
    }
}

function Get-GtsrbIndex {
    param([string]$Root, [string]$ProjectRoot, [int]$SampleLimit)
    $trainRoot = Get-ChildItem -LiteralPath $Root -Directory -Recurse |
        Where-Object { $_.Name -eq 'Training' } |
        Select-Object -First 1
    if ($null -eq $trainRoot) { throw "Diretório Training do GTSRB ausente em $Root" }
    $classNames = @(0..42 | ForEach-Object { [string]$_ })
    $splits = @('train', 'test')
    $rows = @(New-ClassRows -ClassNames $classNames -Splits $splits)
    $samples = New-Object 'System.Collections.Generic.List[string]'
    foreach ($classDir in (Get-ChildItem -LiteralPath $trainRoot.FullName -Directory | Sort-Object -Property Name)) {
        $label = [int]$classDir.Name
        if ($label -lt 0 -or $label -ge $rows.Count) { continue }
        $files = @(
            Get-ChildItem -LiteralPath $classDir.FullName -File -Force |
                Where-Object { $_.Extension -match '^(?i)\.(ppm|png|jpe?g|bmp|gif|tiff?|webp)$' }
        )
        [Int64]$rows[$label].split_train = $files.Count
        [Int64]$rows[$label].total += $files.Count
        $files | Sort-Object -Property Name | Select-Object -First $SampleLimit |
            ForEach-Object { $samples.Add((Convert-ToProjectRelativePath -Path $_.FullName -ProjectRoot $ProjectRoot)) }
    }
    $testCsv = Join-Path $Root 'GT-final_test.csv'
    if (-not (Test-Path -LiteralPath $testCsv -PathType Leaf)) { throw "CSV de rótulos de teste ausente: $testCsv" }
    Import-Csv -LiteralPath $testCsv | Group-Object -Property ClassId | ForEach-Object {
        $label = [int]$_.Name
        if ($label -ge 0 -and $label -lt $rows.Count) {
            [Int64]$rows[$label].split_test = $_.Count
            [Int64]$rows[$label].total += $_.Count
        }
    }
    return [ordered]@{
        class_names = $classNames
        class_distribution = $rows
        original_splits = [ordered]@{
            train = Get-RowSplitTotal -Rows $rows -Split 'train'
            test = Get-RowSplitTotal -Rows $rows -Split 'test'
        }
        sample_paths = @($samples)
        image_inspection_scope = "amostra estratificada de ate $SampleLimit arquivo(s) de treino por classe; contagens de teste do GT-final_test.csv"
    }
}

$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $scriptDirectory '..')).Path
Write-Host '[index] cinic10'
$cinic = Get-Cinic10Index -Root (Join-Path $resolvedDatasetRoot 'cinic10') -ProjectRoot $projectRoot -SampleLimit $SamplePerClass
Write-Host '[index] gtsrb'
$gtsrb = Get-GtsrbIndex -Root (Join-Path $resolvedDatasetRoot 'gtsrb') -ProjectRoot $projectRoot -SampleLimit $SamplePerClass
$payload = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTime]::UtcNow.ToString('o')
    measurement_environment = 'Windows native filesystem enumeration'
    datasets = [ordered]@{ cinic10 = $cinic; gtsrb = $gtsrb }
}
$outputPath = [IO.Path]::GetFullPath($Output)
$null = New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputPath)
[IO.File]::WriteAllText($outputPath, ($payload | ConvertTo-Json -Depth 8) + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
Write-Host "Indice de arquivos salvo em: $outputPath"
