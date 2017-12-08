@echo off

set projectPath=%cd%

echo Removing \Build directory...

rd /s /q "%projectPath%\Build"

echo \Build directory removed (or wasn't there, whatever)!
echo
echo Calling VSMSBuildCmd.bat
echo

call "C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\Tools\VsMSBuildCmd.bat"

cd "%projectPath%"

echo Returning to project directory
echo 

set CMAKE_PREFIX_PATH=C:\xmr-stak-dep\hwloc;C:\xmr-stak-dep\libmicrohttpd;C:\xmr-stak-dep\openssl

mkdir build
cd build

@echo Running cmake VS2017 Win64

cmake -G "Visual Studio 15 2017 Win64" -T v141,host=x64 ..

echo Running build for Release config, Monero currency, and no DCOpenCL

cmake --build . --config Release --target install

echo Copying SSL libraries...

cd bin\Release
xcopy C:\xmr-stak-dep\openssl\bin\*.dll . /v

echo Copy complete!
echo
echo All done!

pause