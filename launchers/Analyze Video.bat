@echo off
REM ===================================================================
REM  Drag a gameplay video onto this file to analyze it.
REM
REM  Two deliberate structural choices, both learned the hard way:
REM
REM  * No `setlocal enabledelayedexpansion`. It mangles any path holding
REM    an exclamation mark, and this script's whole job is handling
REM    paths a user chose.
REM  * Every :label sits at the top level, never inside a parenthesised
REM    if-block. cmd.exe parses a block as a single command, so a label
REM    within one -- or a goto crossing its boundary -- misbehaves.
REM ===================================================================
setlocal
title Gameplay Analyzer

set "HERE=%~dp0"

REM Look in several places, not just beside this file. People move one of
REM the two files, or extract only one; an exe sitting a folder away is
REM not worth a dead end.
set "EXE="
if exist "%HERE%gameplay-analyzer.exe"          set "EXE=%HERE%gameplay-analyzer.exe"
if not defined EXE if exist "%HERE%dist\gameplay-analyzer.exe"  set "EXE=%HERE%dist\gameplay-analyzer.exe"
if not defined EXE if exist "%HERE%..\gameplay-analyzer.exe"    set "EXE=%HERE%..\gameplay-analyzer.exe"
if not defined EXE if exist "%CD%\gameplay-analyzer.exe"        set "EXE=%CD%\gameplay-analyzer.exe"

if defined EXE goto :found

echo.
echo   Could not find gameplay-analyzer.exe
echo.
echo   Looked in:
echo     %HERE%
echo.

REM Windows extracts a file previewed inside a .zip into a temp folder
REM named Temp1_<zipname>, on its own. Double-clicking straight out of the
REM zip is the most common way to land here, so name that before offering
REM generic advice.
echo "%HERE%" | findstr /i /c:"Temp1_" >nul
if not errorlevel 1 goto :inzip
echo "%HERE%" | findstr /i /c:"\AppData\Local\Temp\" >nul
if not errorlevel 1 goto :inzip

echo   The launcher and gameplay-analyzer.exe must be in the same
echo   folder. Move them together and try again.
echo.
pause
exit /b 1

:inzip
echo   ^>^> You are running this from inside the ZIP file. ^<^<
echo.
echo   Windows opened a temporary copy of just this one file, so the
echo   program itself was left behind in the zip.
echo.
echo   To fix it:
echo     1. Find the .zip you downloaded
echo     2. Right-click it and choose "Extract All..."
echo     3. Click "Extract"
echo     4. Open the folder that appears
echo     5. Drag your video onto "Analyze Video" in THAT folder
echo.
pause
exit /b 1

:found
if not "%~1"=="" goto :analyze

echo.
echo   ==========================================================
echo     Gameplay Analyzer
echo   ==========================================================
echo.
echo   To use this: drag a video file ONTO this icon and let go.
echo.
echo   Works with .mp4 and .webm recordings - whatever your capture
echo   software saved, from OBS, ShadowPlay, or the Xbox Game Bar.
echo.
echo   Record at 60fps or higher if you can. Below that, the timing
echo   measurements are not reliable enough to report and the tool
echo   will say so rather than guess.
echo.
echo   ----------------------------------------------------------
echo   First time? Press any key to run a quick self-check and
echo   confirm everything works on this PC.
echo   ----------------------------------------------------------
pause >nul
echo.
"%EXE%" selftest
echo.
pause
exit /b 0

:analyze
set "VIDEO=%~1"
set "NAME=%~n1"
set "OUTDIR=%~dp1%NAME%_analysis"

echo.
echo   Analyzing: %~nx1
echo   Results go to: %OUTDIR%
echo.
echo   This takes a few minutes for a long recording. The window will
echo   stay open when it is done.
echo.

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM Default profile is valorant. Its HUD layout is close enough to most
REM tactical shooters to be a sane starting point; `gameplay-analyzer
REM profiles` lists the alternatives.
"%EXE%" analyze "%VIDEO%" --profile valorant --max-clips 8 ^
        --overlay-dir "%OUTDIR%" --out "%OUTDIR%\results.json"

if errorlevel 1 goto :failed

echo.
echo   ==========================================================
echo     Done. Opening the results folder.
echo   ==========================================================
echo.
echo   The .png images show where your crosshair actually travelled
echo   during each engagement. Red circles mark the moments your aim
echo   changed direction - those are overcorrections.
echo.
echo   results.json has the raw numbers if you want them.
echo.
start "" "%OUTDIR%"
pause
exit /b 0

:failed
echo.
echo   Something went wrong. The message above says what.
echo.
echo   Most common cause: the file is not a video the tool can read.
echo   Try re-saving it as .mp4 and run it again.
echo.
pause
exit /b 1
