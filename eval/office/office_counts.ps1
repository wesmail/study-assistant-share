# Gate item 1 of roadmap item 7: page and slide counts from real Word and PowerPoint.
#
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File office_counts.ps1 -Folder C:\...\copies
#
# Opens every .docx and .pptx in the folder (copies, never the originals) read-only and
# without a window, and prints one JSON line per file. Quits Word or PowerPoint only if
# this script started it. Pages for Word come from ComputeStatistics(wdStatisticPages = 2).
param([Parameter(Mandatory = $true)][string]$Folder)
$ErrorActionPreference = "Stop"

$pptWasRunning = [bool](Get-Process POWERPNT -ErrorAction SilentlyContinue)
$wordWasRunning = [bool](Get-Process WINWORD -ErrorAction SilentlyContinue)
$ppt = $null
$word = $null
try {
    foreach ($file in Get-ChildItem -Path $Folder -File | Sort-Object Name) {
        $row = [ordered]@{ copy = $file.Name }
        try {
            if ($file.Extension -eq ".pptx") {
                if (-not $ppt) { $ppt = New-Object -ComObject PowerPoint.Application }
                # ReadOnly, Untitled, WithWindow (MsoTriState: -1 true, 0 false)
                $deck = $ppt.Presentations.Open($file.FullName, -1, 0, 0)
                $row.pages = $deck.Slides.Count
                $deck.Close()
            } elseif ($file.Extension -eq ".docx") {
                if (-not $word) { $word = New-Object -ComObject Word.Application; $word.Visible = $false }
                $doc = $word.Documents.Open($file.FullName, $false, $true, $false)
                $doc.Repaginate()
                $row.pages = $doc.ComputeStatistics(2)
                $doc.Close(0)
            } else { continue }
        } catch {
            $row.error = $_.Exception.Message
        }
        $row | ConvertTo-Json -Compress
    }
} finally {
    if ($ppt -and -not $pptWasRunning) { $ppt.Quit() }
    if ($word -and -not $wordWasRunning) { $word.Quit() }
}
