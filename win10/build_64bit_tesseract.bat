@rem git submodule update --init --recursive
@rem @if %errorlevel% NEQ 0 GOTO ERROR

reg Query "HKLM\Hardware\Description\System\CentralProcessor\0" | find /i "x86" > NUL && set OS=32BIT || set OS=64BIT
@if %OS%==32BIT call "%programfiles%\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
@if %OS%==64BIT call "%programfiles% (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
@if %errorlevel% NEQ 0 GOTO ERROR

rmdir /S /Q Release64
@if %errorlevel% NEQ 0 GOTO ERROR

msbuild tesseract.sln /p:Configuration=Release /p:Platform=x64 /m
@if %errorlevel% NEQ 0 GOTO ERROR

@rem rmdir /S /Q Debug64
@rem @if %errorlevel% NEQ 0 GOTO ERROR

@rem msbuild tesseract.sln /p:Configuration=Debug /p:Platform=x64 /m
@rem @if %errorlevel% NEQ 0 GOTO ERROR

@GOTO END
:ERROR
@ECHO "Program failed, please check this log file for errors ..." 
@ECHO Errorlevel: %errorlevel%
@EXIT /B %errorlevel%
:END