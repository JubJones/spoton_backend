# SpotOn Backend

After 
```
# Build and start all services
docker-compose up --build -d
# Or (explicit build, then start):
docker-compose build --no-cache
docker-compose up -d

# Stop all services (removes containers, default networks)
docker-compose down

# Restart all services (useful if app code changed and volumes are mounted)
docker-compose restart

# View logs for the backend service
docker-compose logs backend

```
```
docker exec -it -u appuser spoton_backend_service bash -c "dagshub login"
```

# Validate
```
curl -X POST -H "Content-Type: application/json" -d '{"environment_id": "campus"}' http://localhost:8000/api/v1/processing-tasks/start

curl http://localhost:8000/api/v1/processing-tasks/{TASK_ID_FROM_PREVIOUS_STEP}/status

websocat ws://localhost:8000/ws/tracking/{TASK_ID_FROM_PREVIOUS_STEP}
```