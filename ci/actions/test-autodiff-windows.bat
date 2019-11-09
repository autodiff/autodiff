echo --- current directory: %cd% ---

echo === Running tests...
echo call build\tests\tests.exe
call build\tests\tests.exe || goto :error
echo === Running tests...finished!

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
