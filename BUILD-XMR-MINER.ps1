try
{
    Write-Host "`r`nStarting BUILD-XMR Powershell Script`r`n"
    
    Set-ExecutionPolicy Bypass -Scope Process | Out-Null

    Write-Host "Checking required build tools and dependencies are installed..."

    If (!(Get-Module -Listavailable -Name PSCX))
    {
        Write-Host "PowerShell Community Extensions module is installing... " -NoNewline
        Install-Module Pscx -Scope CurrentUser -Force -AllowClobber
        Write-Host "Done." -ForegroundColor Green
    }
    
    If (!(Get-Module -Listavailable -Name VSSetup))
    {
        Write-Host "VSSetup module installing... " -NoNewline
        Install-Module VSSetup -Scope CurrentUser -Force -AllowClobber
        Write-Host "Done." -ForegroundColor Green
    }
        
    $instance = Get-VSSetupInstance | Select-VSSetupInstance -Latest -Require Microsoft.VisualStudio.Component.VC.Tools.x86.x64
    $cmakePackages = $instance.Packages | where Id -Like "*CMake*" | sort Id | select Id,Version

    Write-Host "Checking CMake is installed... " -NoNewline
    
    If (!($cmakePackages))
    {
        Write-Error "CMake is not installed. Please run the Visual Studio Installer and modify your installation to include the CMake tools." -ForegroundColor Red
        Exit
    }

    Write-Host "Done." -ForegroundColor Green

    Write-Host "Importing scripts. " -NoNewline

    . ".\scripts\powershell\Install-CMake.ps1"
    
    Write-Host "Done.`r`n" -ForegroundColor Green

    $path = (Resolve-Path .\).Path
    $currentFolder = "`"$path`""
    $buildPath = "$path\build"
    $releasePath = "$buildPath\bin\release"

    $buildPathExists = Test-Path $buildPath
    $cmakeGenerator = "Visual Studio 15 2017 Win64"
    $cmakeToolset = "v141,host=x64"
    $cmakeArguments = "--build . -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic --config Release --target install -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic"

    $xmrStakDepPath = "C:\xmr-stak-dep"
    $xmrStakDepPathExists = (Test-Path $xmrStakDepPath)

    $hwlocPath = "C:\xmr-stak-dep\hwloc"
    $libmicrohttpdPath = "C:\xmr-stak-dep\libmicrohttpd"
    $openSslPath = "C:\xmr-stak-dep\openssl"
    $sslBinaryPath = "C:\xmr-stak-dep\openssl\bin\*"

    $cmakePrefixPath = "$hwlocPath;$libmicrohttpd;$sslBinaryPath"

    $buildInstructionsUri = "https://github.com/fireice-uk/xmr-stak/blob/dev/doc/compile_Windows.md#dependencies-opensslhwloc-and-microhttpd"

    $msBuildCmdPath = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsMSBuildCmd.bat"
    $msBuildCmdUri = "https://www.visualstudio.com/downloads/"

    $toolsVersion1Uri = "https://github.com/fireice-uk/xmr-stak-dep/releases/download/v1/xmr-stak-dep.zip"
    $toolsVersion2Uri = "https://github.com/fireice-uk/xmr-stak-dep/releases/download/v2/xmr-stak-dep.zip"

    $hwlocPathExists = Test-Path $hwlocPath
    $libmicrohttpdPathExists = Test-Path $libmicrohttpdPath
    $openSslPathExists = Test-Path $openSslPath
    $allDependenciesExist = ($hwlocPathExists -and $libmicrohttpdPathExists -and $openSslPathExists)

    if ($buildPathExists)
    {
        $deleteResponse = Read-Host "`r`nThe '\build' directory aready exists, would you like to delete it and continue? (Y/n)"
        $confirmedDelete = ([System.String]::IsNullOrWhiteSpace($deleteResponse) -or ($deleteResponse -ieq "Y"))
        If (!($confirmedDelete))
        {
            Write-Host "`r`nBuild directory already exists at `"$buildPath`", please delete the directory and try again." -ForegroundColor Red
            Exit
        }
    }

    if (!($xmrStakDepPathExists))
    {
        Write-Host "`r`nThe xmr-stak-dep directory was not found at '$xmrStakDepPath', would you like to download the dependencies and set this up now? (Y/n): " -ForegroundColor Yellow -NoNewline
        $installDepResponse = Read-Host
        $confirmedInstallDep = ([System.String]::IsNullOrWhiteSpace($deleteResponse) -or ($deleteResponse -ieq "Y"))

        If (!($confirmedInstallDep))
        {
            Write-Host "`r`nThe xmr-stak dependencies are not available in '$xmrStakDepPath'. " -NoNewline -ForegroundColor Yellow
            Write-Host "Please download the required dependencies and/or set the correct path before continuing." -ForegroundColor Yellow
        }
    
        $tempFileName = "xmr-stak-dep_" + (New-Guid).ToString() + ".zip"
        $tempFilePath = [System.IO.Path]::GetTempPath() + $tempFileName
        
        Write-Host "`r`nDownloading xmr-stak-dep.zip... " -NoNewline
    
        try
        {
            $client = New-Object System.Net.Webclient
            $client.DownloadFile($toolsVersion2Uri, $tempFilePath)
        }
        catch [System.Net.WebException]
        {
            Write-Host "Failed." -ForegroundColor Red
            Write-Error "`r`nA web exception occured while downloading the file '$toolsVersion2Uri'. $($PSItem.Exception.Message)"
            Exit
        }
        catch [System.Exception]
        {
            Write-Host "Failed." -ForegroundColor Red
            Write-Error "`r`nAn exception occured while downloading the file '$toolsVersion2Uri'. $($PSItem.Exception.Message)"
            Exit
        }

        $extractPath = $xmrStakDepPath.Replace("xmr-stak-dep", "")

        Write-Host "Done." -ForegroundColor Green
        Write-Host "Extracting to '$extractPath'... " -NoNewline

        try
        {
            Expand-Archive -Path $tempFilePath -OutputPath $extractPath
        }
        catch
        {
            Write-Host "Failed." -ForegroundColor Red
            Write-Error "``r`nAn error occured while extracting the file '$toolsVersion2Uri' to '$extractPath'."
            Exit
        }

        Write-Host "Done.`r`n" -ForegroundColor Green
    }

    If (!($allDependenciesExist))
    {
        Write-Host "`r`nMSBuild tools were not found at '$msBuildCmdPath'." -ForegroundColor Red

        Write-Host "`r`nPlease download and install Visual Studio 2017 at $msBuildCmdUri before compiling xmr-stak." -ForegroundColor Red
        Write-Host "$msBuildCmdUri"  -ForegroundColor Red

        Write-Host "`r`nFor more info please read the compile instructions listed here:" -ForegroundColor Yellow
        Write-Host "$buildInstructionsUri" -ForegroundColor Yellow

        Exit
    }

    If (!($allDependenciesExist))
    {
        If (!($hwlocNotPathExists))
        {
            Write-Host "Hwloc was not found at '$hwlocPath'." -ForegroundColor Red
        }

        If (!($libmicrohttpdPathExists))
        {
            Write-Host "Microhttpd was not found at '$libmicrohttpdPath'." -ForegroundColor Red
        }

        If (!($openSslPathExists))
        {
            Write-Host "Open SSL was not found at '$openSslPath'." -ForegroundColor Red
        }

        Write-Host "`r`nPlease download the dependencies and set the path accordingly, you can download the required files from:" -ForegroundColor Red
        Write-Host "Version 1 (CUDA 8) : $toolsVersion1Uri" -ForegroundColor Red
        Write-Host "Version 2 (CUDA 9+): $toolsVersion2Uri" -ForegroundColor Red
        Write-Host "`r`nFor more info please read the compile instructions listed here:" -ForegroundColor Red
        Write-Host "$buildInstructionsUri" -ForegroundColor Red

        Exit
    }

    Write-Host "`r`nBuild started at $(Get-Date -Format T)...`r`n" -ForegroundColor Cyan
    Write-Host "Calling MSBuild command prompt...`r`n"

    # C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsMSBuildCmd.bat
    # Start-Process $msBuildCmdPath -NoNewWindow -Wait
    Invoke-BatchFile -Path $msBuildCmdPath | Out-Host

    Write-Host "`r`nVsMSBuildCmd batch started successfully.`r`n" -ForegroundColor Green

    Write-Host "Setting CMAKE_PREFIX_PATH environment variable..." -NoNewLine
    
    Set-Variable -Name "CMAKE_PREFIX_PATH" -Value "$hwlocPath;$libmicrohttpd;$sslBinaryPath" -Scope Global -Description "CMAKE_PREFIX_PATH" -Force | Out-Null

    Write-Host "Done.`r`n" -ForegroundColor Green

    If (!(Test-Path $buildPath))
    {
        Write-Host "`r`nCreating build directory at '$buildPath'... " -NoNewline
        [system.io.directory]::CreateDirectory($buildPath) | Out-Null
        Write-Host "Build directory created.`r`n" -ForegroundColor Green
    }

    Write-Host "Changing directory to '$buildPath'`r`n"

    Set-Location $buildPath

    Write-Host "Setting CMake generator to '$cmakeGenerator' and toolset to '$cmakeToolset'`r`n"

    # "cmake -G `"Visual Studio 15 2017 Win64`" -T v141,host=x64 .."
    Start-Process cmake -ArgumentList "-G `"$cmakeGenerator`" -T $cmakeToolset .." -NoNewWindow -Wait

    Write-Host "Starting cmake build... " -NoNewline

    # "cmake --build . -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic --config Release --target install -DXMR-STAK_CURRENCY=monero -DOpenSSL_ENABLE=OFF -DXMR-STAK_COMPILE=generic"
    Start-Process cmake -ArgumentList "$cmakeArguments" -Wait -ErrorAction Stop | Write-Output

    Write-Host "Done." -ForegroundColor Green

    If (!(Test-Path $releasePath))
    {
        Write-Error "The build output directory at '$releasePath' does not exist. The build did not run successfully..."
        Exit -1
    }

    Write-Host "`r`nCopying SSL binaries from '$sslBinaryPath' to '$releasePath'... " -NoNewline
    
    Set-Location $releasePath

    Copy-Item -Path $sslBinaryPath -Destination $releasePath | Out-Null

    Write-Host "Done.`r`n" -ForegroundColor Green
    Write-Host "Build completed at $(Get-Date -Format T)" -ForegroundColor Green
}
catch
{
    Write-Host "`r`nAn error occured while building xmr-stak. $($PSItem.Exception.Message)" -ForegroundColor Red

    Exit -1
}
finally
{
    Write-Host ""
    Read-Host "Press ENTER to exit"

    Exit 1
}