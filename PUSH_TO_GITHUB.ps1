# PowerShell script to push Voxzen to GitHub
# Run this script in PowerShell: .\PUSH_TO_GITHUB.ps1

Write-Host "🚀 Voxzen GitHub Push Script" -ForegroundColor Green
Write-Host ""

# Check if git is available
$gitPath = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitPath) {
    Write-Host "❌ Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Or use GitHub Desktop: https://desktop.github.com/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "After installing Git, run this script again." -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Git found: $($gitPath.Source)" -ForegroundColor Green
Write-Host ""

# Check if already a git repo
if (Test-Path .git) {
    Write-Host "⚠️  Git repository already initialized" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Do you want to continue? (y/n)"
    if ($response -ne "y") {
        exit 0
    }
} else {
    Write-Host "Initializing Git repository..." -ForegroundColor Cyan
    git init
    Write-Host "✓ Git repository initialized" -ForegroundColor Green
    Write-Host ""
}

# Check git status
Write-Host "Checking git status..." -ForegroundColor Cyan
git status
Write-Host ""

# Add all files
Write-Host "Adding all files..." -ForegroundColor Cyan
git add .
Write-Host "✓ Files added" -ForegroundColor Green
Write-Host ""

# Commit
Write-Host "Creating commit..." -ForegroundColor Cyan
$commitMessage = "Initial commit - Production-ready Voxzen Hindi→English translation agent with winloop/uvloop optimization"
git commit -m $commitMessage
Write-Host "✓ Commit created" -ForegroundColor Green
Write-Host ""

# Ask for GitHub repository URL
Write-Host "📦 GitHub Repository Setup" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Go to https://github.com/new" -ForegroundColor Yellow
Write-Host "2. Create a new repository (name: voxzen)" -ForegroundColor Yellow
Write-Host "3. Make it Private (recommended)" -ForegroundColor Yellow
Write-Host "4. Don't initialize with README (we already have one)" -ForegroundColor Yellow
Write-Host "5. Copy the repository URL" -ForegroundColor Yellow
Write-Host ""
$repoUrl = Read-Host "Enter your GitHub repository URL (e.g., https://github.com/username/voxzen.git)"

if ($repoUrl) {
    Write-Host ""
    Write-Host "Adding remote origin..." -ForegroundColor Cyan
    git remote add origin $repoUrl 2>$null
    if ($LASTEXITCODE -ne 0) {
        git remote set-url origin $repoUrl
    }
    Write-Host "✓ Remote added" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Setting branch to main..." -ForegroundColor Cyan
    git branch -M main
    Write-Host "✓ Branch set to main" -ForegroundColor Green
    Write-Host ""
    
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan
    git push -u origin main
    Write-Host ""
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Successfully pushed to GitHub!" -ForegroundColor Green
        Write-Host ""
        Write-Host "Repository URL: $repoUrl" -ForegroundColor Cyan
    } else {
        Write-Host "❌ Push failed. Please check the error above." -ForegroundColor Red
        Write-Host ""
        Write-Host "Common issues:" -ForegroundColor Yellow
        Write-Host "  - Authentication required (use GitHub CLI or SSH keys)" -ForegroundColor Yellow
        Write-Host "  - Repository doesn't exist (create it first)" -ForegroundColor Yellow
        Write-Host "  - Wrong URL (check the repository URL)" -ForegroundColor Yellow
    }
} else {
    Write-Host ""
    Write-Host "⚠️  No repository URL provided. Skipping push." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To push later, run:" -ForegroundColor Cyan
    Write-Host "  git remote add origin YOUR_REPO_URL" -ForegroundColor White
    Write-Host "  git push -u origin main" -ForegroundColor White
}

Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "  1. Configure Fly.io secrets (see DEPLOYMENT_GUIDE.md)" -ForegroundColor White
Write-Host "  2. Deploy to Fly.io: fly deploy" -ForegroundColor White
Write-Host "  3. Monitor: fly logs" -ForegroundColor White
Write-Host ""

