#!/bin/bash
echo "🛠️ HACKY BUILD: Forcing build to use HOST NETWORK..."

# This bypasses Docker's internal DNS bridge entirely
docker build --network=host -t spoton_backend_image:gpu -f Dockerfile.gpu .

if [ $? -eq 0 ]; then
    echo "✅ Build Successful! Starting services..."
    # Now start compose, but skip the build step since image is ready
    docker compose -f docker-compose.gpu.yml up
else
    echo "❌ Build Failed even with host network hack."
fi
