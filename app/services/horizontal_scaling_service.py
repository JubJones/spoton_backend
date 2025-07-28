"""
Horizontal scaling service for distributed person tracking.

Handles:
- Load balancing across multiple instances
- Camera allocation and load distribution
- Service discovery and health monitoring
- Failover and recovery mechanisms
- Distributed state synchronization
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
import aioredis
import aiohttp

from app.infrastructure.cache.tracking_cache import tracking_cache
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ServiceInstance:
    """Service instance information."""
    instance_id: str
    host: str
    port: int
    status: str = "healthy"  # healthy, degraded, unhealthy
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    load_score: float = 0.0
    assigned_cameras: Set[str] = field(default_factory=set)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadBalancingConfig:
    """Load balancing configuration."""
    max_cameras_per_instance: int = 4
    load_threshold: float = 0.8
    health_check_interval: int = 10  # seconds
    failover_timeout: int = 30  # seconds
    rebalance_interval: int = 60  # seconds
    heartbeat_timeout: int = 30  # seconds


@dataclass
class ScalingMetrics:
    """Scaling metrics for decision making."""
    timestamp: datetime
    total_instances: int
    healthy_instances: int
    total_cameras: int
    average_load: float
    max_load: float
    min_load: float
    cameras_per_instance: Dict[str, int]
    load_distribution: Dict[str, float]


class HorizontalScalingService:
    """
    Horizontal scaling service for distributed person tracking.
    
    Features:
    - Load balancing across multiple instances
    - Camera allocation and distribution
    - Service discovery and health monitoring
    - Automatic failover and recovery
    - Distributed state synchronization
    """
    
    def __init__(self):
        self.instance_id = self._generate_instance_id()
        self.service_instances: Dict[str, ServiceInstance] = {}
        self.config = LoadBalancingConfig()
        
        # Load balancing state
        self.camera_assignments: Dict[str, str] = {}  # camera_id -> instance_id
        self.load_scores: Dict[str, float] = {}
        self.scaling_metrics: List[ScalingMetrics] = []
        
        # Redis connection for distributed coordination
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.rebalancing_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self.scaling_stats = {
            'instances_registered': 0,
            'camera_assignments': 0,
            'failovers_performed': 0,
            'rebalances_performed': 0,
            'health_checks_completed': 0,
            'load_distributions': 0
        }
        
        logger.info(f"HorizontalScalingService initialized with instance ID: {self.instance_id}")
    
    def _generate_instance_id(self) -> str:
        """Generate unique instance ID."""
        try:
            import socket
            import uuid
            
            hostname = socket.gethostname()
            timestamp = datetime.now(timezone.utc).timestamp()
            unique_id = str(uuid.uuid4())[:8]
            
            return f"{hostname}_{int(timestamp)}_{unique_id}"
            
        except Exception as e:
            logger.error(f"Error generating instance ID: {e}")
            return f"instance_{int(datetime.now(timezone.utc).timestamp())}"
    
    async def initialize(self):
        """Initialize horizontal scaling service."""
        try:
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Register current instance
            await self._register_instance()
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            logger.info("HorizontalScalingService initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing HorizontalScalingService: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup scaling service."""
        try:
            # Stop monitoring
            await self._stop_monitoring()
            
            # Unregister instance
            await self._unregister_instance()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("HorizontalScalingService cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up HorizontalScalingService: {e}")
    
    async def _initialize_redis(self):
        """Initialize Redis connection for distributed coordination."""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            self.redis_client = await aioredis.from_url(redis_url)
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection initialized for scaling service")
            
        except Exception as e:
            logger.error(f"Error initializing Redis connection: {e}")
            raise
    
    # Instance Management
    async def register_instance(
        self,
        host: str,
        port: int,
        capabilities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a service instance."""
        try:
            instance = ServiceInstance(
                instance_id=self.instance_id,
                host=host,
                port=port,
                capabilities=capabilities or {},
                metadata=metadata or {}
            )
            
            self.service_instances[self.instance_id] = instance
            self.scaling_stats['instances_registered'] += 1
            
            # Store in Redis for distributed coordination
            if self.redis_client:
                await self.redis_client.hset(
                    "service_instances",
                    self.instance_id,
                    json.dumps({
                        "host": host,
                        "port": port,
                        "status": "healthy",
                        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
                        "load_score": 0.0,
                        "assigned_cameras": list(instance.assigned_cameras),
                        "capabilities": capabilities or {},
                        "metadata": metadata or {}
                    })
                )
            
            logger.info(f"Registered instance {self.instance_id} at {host}:{port}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering instance: {e}")
            return False
    
    async def _register_instance(self):
        """Register current instance."""
        try:
            host = getattr(settings, 'HOST', 'localhost')
            port = getattr(settings, 'PORT', 8000)
            
            capabilities = {
                'detection': True,
                'reid': True,
                'mapping': True,
                'gpu_enabled': True,
                'max_cameras': self.config.max_cameras_per_instance
            }
            
            metadata = {
                'version': '1.0.0',
                'startup_time': datetime.now(timezone.utc).isoformat()
            }
            
            await self.register_instance(host, port, capabilities, metadata)
            
        except Exception as e:
            logger.error(f"Error registering current instance: {e}")
    
    async def unregister_instance(self, instance_id: str) -> bool:
        """Unregister a service instance."""
        try:
            if instance_id in self.service_instances:
                instance = self.service_instances[instance_id]
                
                # Reassign cameras from this instance
                await self._reassign_cameras(instance_id)
                
                # Remove from local registry
                del self.service_instances[instance_id]
                
                # Remove from Redis
                if self.redis_client:
                    await self.redis_client.hdel("service_instances", instance_id)
                
                logger.info(f"Unregistered instance {instance_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error unregistering instance: {e}")
            return False
    
    async def _unregister_instance(self):
        """Unregister current instance."""
        try:
            await self.unregister_instance(self.instance_id)
            
        except Exception as e:
            logger.error(f"Error unregistering current instance: {e}")
    
    # Load Balancing
    async def assign_camera(self, camera_id: str) -> Optional[str]:
        """Assign camera to best available instance."""
        try:
            # Find instance with lowest load
            best_instance = await self._find_best_instance()
            
            if not best_instance:
                logger.warning(f"No available instances for camera {camera_id}")
                return None
            
            # Assign camera
            self.camera_assignments[camera_id] = best_instance.instance_id
            best_instance.assigned_cameras.add(camera_id)
            
            # Update load score
            await self._update_load_score(best_instance.instance_id)
            
            # Store assignment in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "camera_assignments",
                    camera_id,
                    best_instance.instance_id
                )
            
            self.scaling_stats['camera_assignments'] += 1
            
            logger.info(f"Assigned camera {camera_id} to instance {best_instance.instance_id}")
            return best_instance.instance_id
            
        except Exception as e:
            logger.error(f"Error assigning camera: {e}")
            return None
    
    async def _find_best_instance(self) -> Optional[ServiceInstance]:
        """Find best available instance for camera assignment."""
        try:
            available_instances = []
            
            for instance in self.service_instances.values():
                if (instance.status == "healthy" and 
                    len(instance.assigned_cameras) < self.config.max_cameras_per_instance and
                    instance.load_score < self.config.load_threshold):
                    available_instances.append(instance)
            
            if not available_instances:
                return None
            
            # Sort by load score (ascending)
            available_instances.sort(key=lambda x: x.load_score)
            
            return available_instances[0]
            
        except Exception as e:
            logger.error(f"Error finding best instance: {e}")
            return None
    
    async def _reassign_cameras(self, failed_instance_id: str):
        """Reassign cameras from failed instance."""
        try:
            cameras_to_reassign = []
            
            # Find cameras assigned to failed instance
            for camera_id, instance_id in self.camera_assignments.items():
                if instance_id == failed_instance_id:
                    cameras_to_reassign.append(camera_id)
            
            # Reassign each camera
            for camera_id in cameras_to_reassign:
                new_instance_id = await self.assign_camera(camera_id)
                if new_instance_id:
                    logger.info(f"Reassigned camera {camera_id} from {failed_instance_id} to {new_instance_id}")
                    self.scaling_stats['failovers_performed'] += 1
                else:
                    logger.error(f"Failed to reassign camera {camera_id}")
            
        except Exception as e:
            logger.error(f"Error reassigning cameras: {e}")
    
    async def _update_load_score(self, instance_id: str):
        """Update load score for instance."""
        try:
            if instance_id not in self.service_instances:
                return
            
            instance = self.service_instances[instance_id]
            
            # Calculate load score based on assigned cameras and capabilities
            camera_load = len(instance.assigned_cameras) / self.config.max_cameras_per_instance
            
            # Additional factors could include:
            # - CPU usage
            # - Memory usage
            # - GPU utilization
            # - Network bandwidth
            
            instance.load_score = camera_load
            self.load_scores[instance_id] = camera_load
            
            # Update in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "instance_loads",
                    instance_id,
                    str(camera_load)
                )
            
        except Exception as e:
            logger.error(f"Error updating load score: {e}")
    
    # Health Monitoring
    async def _start_monitoring(self):
        """Start monitoring tasks."""
        try:
            self.monitoring_active = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.health_check_task = asyncio.create_task(self._health_check_loop())
            self.rebalancing_task = asyncio.create_task(self._rebalancing_loop())
            
            logger.info("Monitoring tasks started")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {e}")
    
    async def _stop_monitoring(self):
        """Stop monitoring tasks."""
        try:
            self.monitoring_active = False
            
            tasks = [self.monitoring_task, self.health_check_task, self.rebalancing_task]
            for task in tasks:
                if task:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            logger.info("Monitoring tasks stopped")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                await self._collect_scaling_metrics()
                await asyncio.sleep(10)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _health_check_loop(self):
        """Health check loop."""
        try:
            while self.monitoring_active:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("Health check loop cancelled")
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _rebalancing_loop(self):
        """Load rebalancing loop."""
        try:
            while self.monitoring_active:
                await self._perform_load_rebalancing()
                await asyncio.sleep(self.config.rebalance_interval)
                
        except asyncio.CancelledError:
            logger.info("Rebalancing loop cancelled")
        except Exception as e:
            logger.error(f"Error in rebalancing loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all instances."""
        try:
            current_time = datetime.now(timezone.utc)
            
            for instance_id, instance in list(self.service_instances.items()):
                # Check heartbeat timeout
                if (current_time - instance.last_heartbeat).total_seconds() > self.config.heartbeat_timeout:
                    logger.warning(f"Instance {instance_id} heartbeat timeout")
                    instance.status = "unhealthy"
                    
                    # Reassign cameras
                    await self._reassign_cameras(instance_id)
                    
                    # Remove unhealthy instance
                    await self.unregister_instance(instance_id)
                    
                else:
                    # Perform HTTP health check
                    health_status = await self._check_instance_health(instance)
                    instance.status = health_status
                    
                    if health_status == "unhealthy":
                        await self._reassign_cameras(instance_id)
            
            self.scaling_stats['health_checks_completed'] += 1
            
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
    
    async def _check_instance_health(self, instance: ServiceInstance) -> str:
        """Check health of specific instance."""
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"http://{instance.host}:{instance.port}/health"
                
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return "healthy"
                    else:
                        return "degraded"
                        
        except Exception as e:
            logger.error(f"Error checking instance health: {e}")
            return "unhealthy"
    
    async def _perform_load_rebalancing(self):
        """Perform load rebalancing across instances."""
        try:
            if len(self.service_instances) < 2:
                return
            
            # Calculate load distribution
            load_distribution = {}
            for instance_id, instance in self.service_instances.items():
                load_distribution[instance_id] = len(instance.assigned_cameras)
            
            # Find imbalanced instances
            max_load = max(load_distribution.values())
            min_load = min(load_distribution.values())
            
            # Rebalance if difference is significant
            if max_load - min_load > 1:
                await self._rebalance_cameras(load_distribution)
                self.scaling_stats['rebalances_performed'] += 1
            
        except Exception as e:
            logger.error(f"Error performing load rebalancing: {e}")
    
    async def _rebalance_cameras(self, load_distribution: Dict[str, int]):
        """Rebalance cameras across instances."""
        try:
            # Find overloaded and underloaded instances
            sorted_instances = sorted(load_distribution.items(), key=lambda x: x[1])
            
            underloaded = sorted_instances[:len(sorted_instances)//2]
            overloaded = sorted_instances[len(sorted_instances)//2:]
            
            # Move cameras from overloaded to underloaded instances
            for overloaded_id, _ in overloaded:
                for underloaded_id, _ in underloaded:
                    if load_distribution[overloaded_id] > load_distribution[underloaded_id] + 1:
                        # Move one camera
                        await self._move_camera(overloaded_id, underloaded_id)
                        load_distribution[overloaded_id] -= 1
                        load_distribution[underloaded_id] += 1
                        break
            
        except Exception as e:
            logger.error(f"Error rebalancing cameras: {e}")
    
    async def _move_camera(self, from_instance_id: str, to_instance_id: str):
        """Move camera from one instance to another."""
        try:
            from_instance = self.service_instances.get(from_instance_id)
            to_instance = self.service_instances.get(to_instance_id)
            
            if not from_instance or not to_instance:
                return
            
            # Find camera to move
            if from_instance.assigned_cameras:
                camera_id = list(from_instance.assigned_cameras)[0]
                
                # Update assignments
                from_instance.assigned_cameras.remove(camera_id)
                to_instance.assigned_cameras.add(camera_id)
                self.camera_assignments[camera_id] = to_instance_id
                
                # Update load scores
                await self._update_load_score(from_instance_id)
                await self._update_load_score(to_instance_id)
                
                # Update Redis
                if self.redis_client:
                    await self.redis_client.hset(
                        "camera_assignments",
                        camera_id,
                        to_instance_id
                    )
                
                logger.info(f"Moved camera {camera_id} from {from_instance_id} to {to_instance_id}")
            
        except Exception as e:
            logger.error(f"Error moving camera: {e}")
    
    async def _collect_scaling_metrics(self):
        """Collect scaling metrics."""
        try:
            current_time = datetime.now(timezone.utc)
            
            healthy_instances = sum(1 for i in self.service_instances.values() if i.status == "healthy")
            total_cameras = len(self.camera_assignments)
            
            load_scores = [i.load_score for i in self.service_instances.values()]
            
            cameras_per_instance = {
                instance_id: len(instance.assigned_cameras)
                for instance_id, instance in self.service_instances.items()
            }
            
            load_distribution = {
                instance_id: instance.load_score
                for instance_id, instance in self.service_instances.items()
            }
            
            metrics = ScalingMetrics(
                timestamp=current_time,
                total_instances=len(self.service_instances),
                healthy_instances=healthy_instances,
                total_cameras=total_cameras,
                average_load=sum(load_scores) / len(load_scores) if load_scores else 0.0,
                max_load=max(load_scores) if load_scores else 0.0,
                min_load=min(load_scores) if load_scores else 0.0,
                cameras_per_instance=cameras_per_instance,
                load_distribution=load_distribution
            )
            
            self.scaling_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.scaling_metrics) > 1000:
                self.scaling_metrics = self.scaling_metrics[-1000:]
            
        except Exception as e:
            logger.error(f"Error collecting scaling metrics: {e}")
    
    # API Methods
    async def get_service_instances(self) -> List[Dict[str, Any]]:
        """Get all service instances."""
        try:
            instances = []
            for instance in self.service_instances.values():
                instances.append({
                    'instance_id': instance.instance_id,
                    'host': instance.host,
                    'port': instance.port,
                    'status': instance.status,
                    'last_heartbeat': instance.last_heartbeat.isoformat(),
                    'load_score': instance.load_score,
                    'assigned_cameras': list(instance.assigned_cameras),
                    'capabilities': instance.capabilities,
                    'metadata': instance.metadata
                })
            return instances
            
        except Exception as e:
            logger.error(f"Error getting service instances: {e}")
            return []
    
    async def get_camera_assignments(self) -> Dict[str, str]:
        """Get current camera assignments."""
        return self.camera_assignments.copy()
    
    async def get_scaling_metrics(self) -> List[Dict[str, Any]]:
        """Get scaling metrics."""
        try:
            metrics = []
            for metric in self.scaling_metrics[-100:]:  # Last 100 metrics
                metrics.append({
                    'timestamp': metric.timestamp.isoformat(),
                    'total_instances': metric.total_instances,
                    'healthy_instances': metric.healthy_instances,
                    'total_cameras': metric.total_cameras,
                    'average_load': metric.average_load,
                    'max_load': metric.max_load,
                    'min_load': metric.min_load,
                    'cameras_per_instance': metric.cameras_per_instance,
                    'load_distribution': metric.load_distribution
                })
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting scaling metrics: {e}")
            return []
    
    async def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        try:
            return {
                'scaling_stats': self.scaling_stats,
                'current_instance_id': self.instance_id,
                'total_instances': len(self.service_instances),
                'healthy_instances': sum(1 for i in self.service_instances.values() if i.status == "healthy"),
                'total_cameras_assigned': len(self.camera_assignments),
                'load_balancing_config': {
                    'max_cameras_per_instance': self.config.max_cameras_per_instance,
                    'load_threshold': self.config.load_threshold,
                    'health_check_interval': self.config.health_check_interval
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting scaling statistics: {e}")
            return {}
    
    def reset_statistics(self):
        """Reset scaling statistics."""
        self.scaling_stats = {
            'instances_registered': 0,
            'camera_assignments': 0,
            'failovers_performed': 0,
            'rebalances_performed': 0,
            'health_checks_completed': 0,
            'load_distributions': 0
        }
        logger.info("Scaling statistics reset")


# Global horizontal scaling service instance
horizontal_scaling_service = HorizontalScalingService()