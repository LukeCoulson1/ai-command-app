@echo off
setlocal

REM Prompt for GitHub repository URL
set /p GITHUB_URL=Enter your GitHub repository URL (e.g. https://github.com/username/repo.git): 

REM Prompt for commit message
set /p COMMIT_MSG=Enter your commit message: 

REM Initialize git repo if not already initialized
if not exist ".git" (
    git init
)

REM Add all files
git add .

REM Commit changes
git commit -m "%COMMIT_MSG%"

REM Set remote origin (overwrite if exists)
git remote remove origin 2>nul
git remote add origin %GITHUB_URL%

REM Push to GitHub (main branch)
git branch -M main
git push -u origin main

pause