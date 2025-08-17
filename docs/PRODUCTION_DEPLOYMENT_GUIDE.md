# SpotOn Backend - Production Deployment Guide

**Status**: Production Ready  
**Phase**: 11 - Final Production Enablement ✅ Complete  
**Last Updated**: 2024

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env.production
nano .env.production  # Configure production settings

# 2. Enable production mode
export PRODUCTION_MODE=true

# 3. Start with Docker
docker-compose -f docker-compose.prod.yml up -d

# 4. Verify deployment
curl https://your-api-domain.com/health
```

## Environment Configuration

### Production Environment Variables

Create `.env.production` with the following configuration:

```bash
# --- Application Settings ---
APP_NAME="SpotOn Backend"
API_V1_PREFIX="/api/v1"
DEBUG=false

# --- Production Endpoint Control ---
ENABLE_ANALYTICS_ENDPOINTS=true
ENABLE_EXPORT_ENDPOINTS=true
ENABLE_AUTH_ENDPOINTS=true
ENABLE_ADMIN_ENDPOINTS=false  # Enable only if admin interface needed
PRODUCTION_MODE=true

# --- Security Configuration ---
# CRITICAL: Replace with your actual production URLs
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://your-admin-domain.com
ALLOWED_HOSTS=your-api-domain.com,api.example.com
RATE_LIMIT_PER_MINUTE=30  # More restrictive for production
ENABLE_SECURITY_HEADERS=true
ENABLE_REQUEST_LOGGING=true
MAX_REQUEST_SIZE_MB=5

# --- Database Configuration ---
POSTGRES_USER=spoton_prod_user
POSTGRES_PASSWORD=your-secure-database-password
POSTGRES_SERVER=your-db-host
POSTGRES_PORT=5432
POSTGRES_DB=spoton_production
DATABASE_URL=postgresql://spoton_prod_user:password@your-db-host:5432/spoton_production

# --- Redis Configuration ---
REDIS_HOST=your-redis-host
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your-redis-password

# --- S3 Storage Configuration ---
S3_ENDPOINT_URL=https://s3.amazonaws.com  # or your S3-compatible service
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=spoton-production

# --- JWT & Encryption Configuration ---
JWT_SECRET_KEY=your-very-secure-jwt-secret-key-here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
ENCRYPTION_SECRET_KEY=your-encryption-key-here

# --- AI Model Configuration ---
DETECTOR_TYPE=fasterrcnn
TRACKER_TYPE=botsort
REID_MODEL_TYPE=clip
WEIGHTS_DIR=/app/weights
```

### Security Considerations

**CORS Configuration**:
```bash
# ❌ NEVER use in production
ALLOWED_ORIGINS=*

# ✅ Always specify exact domains
ALLOWED_ORIGINS=https://app.yourdomain.com,https://admin.yourdomain.com
```

**Rate Limiting**:
```bash
# Production recommended settings
RATE_LIMIT_PER_MINUTE=30      # Adjust based on expected load
MAX_REQUEST_SIZE_MB=5         # Smaller limit for production
```

**Host Validation**:
```bash
# Include all valid hostnames for your API
ALLOWED_HOSTS=api.yourdomain.com,your-api-server.com,load-balancer-dns.com
```

## Deployment Options

### Option 1: Docker Compose (Recommended)

1. **Create production Docker Compose file**:
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  backend:
    build: 
      context: .
      dockerfile: Dockerfile
    env_file:
      - .env.production
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: timescale/timescaledb:latest-pg14
    env_file:
      - .env.production  
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
    volumes:
      - redis_data:/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

2. **Deploy**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Option 2: Kubernetes

1. **Create Kubernetes manifests** (see `k8s/` directory)
2. **Deploy**:
```bash
kubectl apply -f k8s/
```

### Option 3: Cloud Services

**AWS ECS/Fargate**:
- Use provided ECS task definitions
- Configure Application Load Balancer
- Set up RDS for PostgreSQL and ElastiCache for Redis

**Google Cloud Run**:
- Deploy using provided Cloud Build configuration
- Configure Cloud SQL and Memorystore

## SSL/TLS Configuration

### Nginx Configuration

Create `nginx.conf`:
```nginx
upstream backend {
    server backend:8000;
}

server {
    listen 80;
    server_name your-api-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-api-domain.com;

    ssl_certificate /etc/ssl/certs/your-domain.crt;
    ssl_certificate_key /etc/ssl/certs/your-domain.key;
    
    # Security headers (additional layer)
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;
    
    location / {
        limit_req zone=api burst=10 nodelay;
        
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
    
    # Health check endpoint (bypass rate limiting)
    location /health {
        proxy_pass http://backend;
        proxy_set_header Host $host;
    }
}
```

## Health Monitoring Setup

### Health Check Endpoints

**Basic Health**:
```bash
curl https://your-api-domain.com/health
```

**System Performance**:
```bash
curl https://your-api-domain.com/api/v1/system/performance/metrics
```

**Comprehensive Health**:
```bash
curl https://your-api-domain.com/api/v1/system/health/comprehensive
```

### Monitoring Integration

**Prometheus Metrics** (if available):
```bash
# Add to your monitoring stack
- job_name: 'spoton-backend'
  static_configs:
    - targets: ['your-api-domain.com:8000']
  metrics_path: '/metrics'  # If metrics endpoint available
```

**Custom Monitoring**:
```bash
# Monitor critical endpoints
*/5 * * * * curl -f https://your-api-domain.com/health || alert
*/1 * * * * curl -s https://your-api-domain.com/api/v1/system/performance/metrics | jq '.cpu_usage_percent' | awk '$1 > 90 {exit 1}'
```

## Performance Tuning

### Application Settings

**Production Performance Configuration**:
```bash
# Uvicorn settings for production
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --access-log \
  --log-level info
```

**Resource Limits**:
```yaml
# In docker-compose.prod.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'  
          memory: 2G
```

### Database Optimization

**PostgreSQL Configuration**:
```sql
-- Recommended production settings
shared_preload_libraries = 'timescaledb'
max_connections = 200
shared_buffers = 1GB
effective_cache_size = 3GB
work_mem = 50MB
maintenance_work_mem = 512MB
```

**Connection Pooling**:
```bash
# Consider using PgBouncer
POSTGRES_URL=postgresql://user:pass@pgbouncer:6432/spoton_production
```

## Security Hardening

### Firewall Configuration

**Allow only necessary ports**:
```bash
# ufw example
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block direct API access
ufw enable
```

### Secrets Management

**Using Docker Secrets**:
```yaml
services:
  backend:
    secrets:
      - db_password
      - jwt_secret
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      JWT_SECRET_KEY_FILE: /run/secrets/jwt_secret

secrets:
  db_password:
    file: ./secrets/db_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt
```

**Using Kubernetes Secrets**:
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: spoton-secrets
type: Opaque
stringData:
  postgres-password: your-secure-password
  jwt-secret: your-jwt-secret
```

## Backup and Recovery

### Database Backup

**Automated Backup Script**:
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump postgresql://user:pass@host:5432/spoton_production > backup_$DATE.sql
aws s3 cp backup_$DATE.sql s3://your-backup-bucket/database/
```

**Cron Schedule**:
```bash
# Daily backups at 2 AM
0 2 * * * /path/to/backup.sh
```

### Application Data Backup

**Redis Backup**:
```bash
# Redis backup
redis-cli --rdb backup.rdb
aws s3 cp backup.rdb s3://your-backup-bucket/redis/
```

## Troubleshooting

### Common Issues

**1. CORS Errors**:
```bash
# Check ALLOWED_ORIGINS configuration
curl -H "Origin: https://your-frontend.com" -I https://your-api.com/health
```

**2. Rate Limiting Issues**:
```bash
# Check rate limit headers
curl -I https://your-api.com/health | grep X-RateLimit
```

**3. Database Connection Issues**:
```bash
# Test database connectivity
docker exec -it container_name psql -h postgres -U spoton_user -d spoton_production
```

### Logs Analysis

**View Application Logs**:
```bash
# Docker Compose
docker-compose logs -f backend

# Kubernetes  
kubectl logs -f deployment/spoton-backend
```

**Monitor System Metrics**:
```bash
# Real-time system monitoring
curl -s https://your-api.com/api/v1/system/performance/metrics | jq
```

## Scaling Considerations

### Horizontal Scaling

**Load Balancer Configuration**:
```nginx
upstream backend_pool {
    least_conn;
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

**Session Affinity**:
```bash
# Use Redis for session storage to support multiple instances
REDIS_SESSION_STORE=true
```

### Vertical Scaling

**Resource Monitoring**:
```bash
# Monitor resource usage
curl -s https://your-api.com/api/v1/system/performance/dashboard | jq '.current_metrics'
```

**Auto-scaling Triggers**:
- CPU usage > 70%
- Memory usage > 80%
- Response time > 1s
- Error rate > 5%

## Maintenance

### Regular Maintenance Tasks

**Weekly**:
```bash
# Clear old logs
docker-compose exec backend python -c "from app.services.memory_manager import memory_manager; memory_manager.cleanup_resources()"

# Garbage collection
curl -X POST https://your-api.com/api/v1/system/maintenance/garbage-collect
```

**Monthly**:
```bash
# Database maintenance
docker-compose exec postgres vacuumdb -U spoton_user spoton_production

# Clear cache
curl -X POST https://your-api.com/api/v1/system/maintenance/clear-cache
```

### Updates and Patches

**Rolling Update Process**:
1. Test in staging environment
2. Create database backup
3. Deploy new version
4. Verify health checks
5. Monitor error rates

**Zero-Downtime Deployment**:
```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up -d --force-recreate --no-deps backend
```

## Compliance and Auditing

### Security Audit Checklist

- [ ] All secrets stored securely (not in code)
- [ ] HTTPS enabled with strong SSL configuration
- [ ] Rate limiting configured appropriately  
- [ ] CORS origins restricted to production domains
- [ ] Security headers enabled
- [ ] Database connections encrypted
- [ ] Regular security updates applied
- [ ] Access logs monitored for suspicious activity

### Performance Audit

- [ ] Response times within SLA (< 1s for most endpoints)
- [ ] Resource usage monitored and within limits
- [ ] Database queries optimized
- [ ] Caching strategy implemented
- [ ] Load testing completed successfully

### Compliance Requirements

**Data Protection**:
- GDPR compliance for EU users
- Data retention policies implemented
- User data export/deletion capabilities
- Audit trail for data access

**Industry Standards**:
- SOC 2 Type II compliance
- ISO 27001 security standards
- OWASP security guidelines

## Support and Monitoring

### Production Support Contacts

**Technical Issues**:
- Primary: DevOps Team
- Secondary: Backend Development Team
- Escalation: Architecture Team

**Monitoring Dashboards**:
- Application Performance: `/api/v1/system/performance/dashboard`
- System Health: `/api/v1/system/health/comprehensive`
- Security Alerts: `/api/v1/system/alerts`

**Emergency Procedures**:
1. Check system health endpoints
2. Review application logs
3. Verify database connectivity
4. Check external service dependencies
5. Escalate to on-call engineer

---

**🎉 Congratulations! Your SpotOn backend is now production-ready with comprehensive monitoring, security hardening, and performance optimization.**