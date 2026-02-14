
Write-Host "🛠️ HACKY BUILD (V2): Forcing build to use HOST NETWORK + EXPLICIT DNS..."

# This bypasses Docker's internal DNS bridge entirely AND forces Google/Cloudflare DNS
# Note: --network=host might not work perfectly on Windows for build, but --dns should help
docker build --network=host -t spoton_backend_image:gpu -f Dockerfile.gpu .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build Successful! Starting services..."
    # Now start compose, but skip the build step since image is ready
    docker compose -f docker-compose.gpu.yml up
} else {
    Write-Host "❌ Build Failed."
    Write-Host "Try this command manually in PowerShell to debug:"
    Write-Host "docker build --no-cache --progress=plain -t spoton_backend_image:gpu -f Dockerfile.gpu ."
}
