@echo off
REM ===================================================================
REM  Drag a gameplay video onto this file to analyze it.
REM
REM  Written for someone who has never opened a terminal, so it does
REM  the two things a bare CLI does not: it explains what to do when
REM  run with no input (double-clicked, which is the obvious first
REM  thing to try), and it stays open at the end so the results can
REM  actually be read instead of flashing past.
REM ===================================================================
setlocal enabledelayedexpansion
title Gameplay Analyzer

set "HERE=%~dp0"
set "EXE=%HERE%gameplay-analyzer.exe"

if not exist "%EXE%" (
  echo.
  echo   Could not find gameplay-analyzer.exe
  echo.
  echo   This launcher needs to sit in the SAME FOLDER as the
  echo   gameplay-analyzer.exe file. Move them together and try again.
  echo.
  pause
  exit /b 1
)

REM %~1 is the dropped file, with surrounding quotes stripped.
if "%~1"=="" (
  echo.
  echo   ==========================================================
  echo     Gameplay Analyzer
  echo   ==========================================================
  echo.
  echo   To use this: drag a video file ONTO this icon and let go.
  echo.
  echo   Works with .mp4 and .webm recordings - whatever your
  echo   capture software saved, from OBS, ShadowPlay, or the
  echo   Xbox Game Bar.
  echo.
  echo   Record at 60fps or higher if you can. Below that, the
  echo   timing measurements are not reliable enough to report
  echo   and the tool will say so rather than guess.
  echo.
  echo   ----------------------------------------------------------
  echo   First time? Press any key to run a quick self-check
  echo   and confirm everything works on this PC.
  echo   ----------------------------------------------------------
  pause >nul
  echo.
  "%EXE%" selftest
  echo.
  pause
  exit /b 0
)

set "VIDEO=%~1"
set "NAME=%~n1"
set "OUTDIR=%~dp1%NAME%_analysis"

echo.
echo   Analyzing: %~nx1
echo   Results go to: %OUTDIR%
echo.
echo   This takes a few minutes for a long recording. The window
echo   will stay open when it is done.
echo.

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

REM Default profile is valorant. Its HUD layout is close enough to most
REM tactical shooters to be a sane starting point; `gameplay-analyzer
REM profiles` lists the alternatives.
"%EXE%" analyze "%VIDEO%" --profile valorant --max-clips 8 ^
        --overlay-dir "%OUTDIR%" --out "%OUTDIR%\results.json"

if errorlevel 1 (
  echo.
  echo   Something went wrong. The message above says what.
  echo.
  echo   Most common cause: the file is not a video the tool can
  echo   read. Try re-saving it as .mp4 and run it again.
  echo.
  pause
  exit /b 1
)

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
