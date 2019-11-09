echo === Running tests... || goto :error
echo call build\tests\%CONFIGURATION%\tests.exe || goto :error
call build\tests\%CONFIGURATION%\tests.exe || goto :error
echo === Running tests...finished! || goto :error

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
