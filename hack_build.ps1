
Write-Host "🛠️ HACKY BUILD (V3): Disabling BuildKit and using Host Network..."

# BuildKit (default) often fails with --network=host on Windows due to different VM handling.
# Legacy builder is more reliable for this specific issue.
$env:DOCKER_BUILDKIT=0

# Standard build command with network=host (DNS flag removed as it's invalid for build)
docker build --network=host -t spoton_backend_image:gpu -f Dockerfile.gpu .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Build Successful! Starting services..."
    # Now start compose, but skip the build step since image is ready
    docker compose -f docker-compose.gpu.yml up
} else {
    Write-Host "❌ Build Failed."
}
