# Need param block here

$external_counter = 0
$sw = [Diagnostics.Stopwatch]::StartNew()
# Recurse through folder
get-childitem $InputFileFullPath -recurse | where {$_.extension -eq ".XXXYYY"} | % {
	$outName = "out-"+$rand+"-"+$external_counter+".XXXYYY"
	$inName = $_.FullName
	Start-Process <someprocess>.exe -ArgumentList `"$inName`",$outName -NoNewWindow -Wait
	$external_counter ++
}

$sw.Stop()
$totalSec = $sw.Elapsed.TotalSeconds

Write-Output "Elapsed Time = [$totalSec] seconds for [$external_counter] files"