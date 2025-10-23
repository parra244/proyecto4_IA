# Script para configurar Git y subir el proyecto a GitHub
# Ejecutar después de instalar Git

Write-Host "=== Configuración de Git para Proyecto IA ===" -ForegroundColor Green

# Verificar si Git está instalado
try {
    git --version
    Write-Host "Git está instalado correctamente" -ForegroundColor Green
} catch {
    Write-Host "Error: Git no está instalado. Por favor instala Git desde https://git-scm.com/download/win" -ForegroundColor Red
    exit 1
}

# Inicializar repositorio Git
Write-Host "`nInicializando repositorio Git..." -ForegroundColor Yellow
git init

# Configurar usuario (reemplaza con tus datos)
Write-Host "`nConfigurando usuario de Git..." -ForegroundColor Yellow
$gitUser = Read-Host "Ingresa tu nombre de usuario de Git"
$gitEmail = Read-Host "Ingresa tu email de Git"

git config user.name $gitUser
git config user.email $gitEmail

# Agregar archivos al staging
Write-Host "`nAgregando archivos al staging area..." -ForegroundColor Yellow
git add .

# Hacer commit inicial
Write-Host "`nHaciendo commit inicial..." -ForegroundColor Yellow
git commit -m "Commit inicial: Proyecto IA para predicción de deserción laboral"

# Configurar repositorio remoto
Write-Host "`nConfigurando repositorio remoto..." -ForegroundColor Yellow
$repoUrl = Read-Host "Ingresa la URL de tu repositorio de GitHub (ej: https://github.com/tuusuario/proyectoIA.git)"

git remote add origin $repoUrl

# Verificar rama principal
$branch = git branch --show-current
if (-not $branch) {
    git branch -M main
    $branch = "main"
}

# Subir a GitHub
Write-Host "`nSubiendo código a GitHub..." -ForegroundColor Yellow
git push -u origin $branch

Write-Host "`n=== ¡Proyecto subido exitosamente a GitHub! ===" -ForegroundColor Green
Write-Host "Tu proyecto está disponible en: $repoUrl" -ForegroundColor Cyan

