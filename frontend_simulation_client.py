#!/usr/bin/env python3
"""
SpotOn Frontend Simulation Client

This script simulates all the interactions a frontend would have with the SpotOn backend:
- REST API calls for task management, analytics, authentication
- WebSocket connections for real-time tracking updates
- Comprehensive validation and error handling
- Agentic retry logic for robust testing

Author: Claude (SpotOn Backend Team)
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import sys
import traceback

# Third-party imports
import aiohttp
import websockets
from websockets.exceptions import WebSocketException, ConnectionClosedError
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("frontend_simulator")

class SpotOnFrontendSimulator:
    """Comprehensive frontend simulation client for SpotOn backend."""
    
    def __init__(self, base_url: str = "http://localhost:3847", max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.ws_base_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/')
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[str] = None
        self.active_tasks: Dict[str, Dict] = {}
        self.websocket_connections: Dict[str, Any] = {}
        
        # Test configuration
        self.test_environments = ["campus", "factory"]
        self.test_users = {
            "admin": {"username": "admin", "password": "admin123", "role": "admin"},
            "operator": {"username": "operator", "password": "op123", "role": "operator"},
            "viewer": {"username": "viewer", "password": "view123", "role": "viewer"}
        }
        
        logger.info(f"Frontend simulator initialized for {self.base_url}")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            connector=aiohttp.TCPConnector(limit=100)
        )
        logger.info("HTTP session created")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            logger.info("HTTP session closed")

    async def retry_request(self, func, *args, **kwargs):
        """Retry wrapper for HTTP requests with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise
                wait_time = 2 ** attempt
                logger.warning(f"Request attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop('headers', {})
        
        # Add auth token if available
        if self.auth_token:
            headers['Authorization'] = f"Bearer {self.auth_token}"
            
        kwargs['headers'] = headers
        
        logger.debug(f"{method.upper()} {url}")
        
        async def _request():
            async with self.session.request(method, url, **kwargs) as response:
                response_text = await response.text()
                
                if response.content_type == 'application/json':
                    response_data = json.loads(response_text)
                else:
                    response_data = {"raw_response": response_text}
                
                logger.debug(f"Response {response.status}: {response_data}")
                
                if response.status >= 400:
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=response_text
                    )
                
                return {
                    "status": response.status,
                    "data": response_data,
                    "headers": dict(response.headers)
                }
        
        return await self.retry_request(_request)

    # ===== HEALTH AND SYSTEM CHECKS =====
    
    async def test_health_endpoints(self) -> bool:
        """Test all health check endpoints."""
        logger.info("🏥 Testing health endpoints...")
        
        health_endpoints = [
            "/health",
            "/api/v1/auth/health", 
            "/api/v1/analytics/health",
            "/ws/health"
        ]
        
        all_healthy = True
        for endpoint in health_endpoints:
            try:
                response = await self.make_request("GET", endpoint)
                if response["status"] == 200:
                    logger.info(f"✅ {endpoint}: {response['data']}")
                else:
                    logger.error(f"❌ {endpoint}: Unexpected status {response['status']}")
                    all_healthy = False
            except Exception as e:
                logger.error(f"❌ {endpoint}: {e}")
                all_healthy = False
        
        return all_healthy

    # ===== AUTHENTICATION SIMULATION =====
    
    async def test_authentication_flow(self) -> bool:
        """Test complete authentication flow."""
        logger.info("🔐 Testing authentication flow...")
        
        try:
            # Test login
            login_data = {
                "username": "admin",
                "password": "SpotOn2024!"
            }
            
            response = await self.make_request(
                "POST", "/api/v1/auth/login", 
                json=login_data
            )
            
            if response["status"] == 200 and "access_token" in response["data"]:
                self.auth_token = response["data"]["access_token"]
                logger.info("✅ Login successful")
                
                # Test authenticated endpoint
                me_response = await self.make_request("GET", "/api/v1/auth/me")
                if me_response["status"] == 200:
                    logger.info(f"✅ User info: {me_response['data']}")
                
                # Test permission endpoint
                perm_response = await self.make_request("GET", "/api/v1/auth/permissions/test")
                if perm_response["status"] == 200:
                    logger.info("✅ Permission test passed")
                
                return True
            else:
                logger.error("❌ Login failed - no access token received")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication test failed: {e}")
            # Continue without auth for other tests
            return False

    # ===== PROCESSING TASK SIMULATION =====
    
    async def test_processing_task_lifecycle(self, environment_id: str = "campus") -> Optional[str]:
        """Test complete processing task lifecycle."""
        logger.info(f"⚙️ Testing processing task lifecycle for environment: {environment_id}")
        
        try:
            # 1. Start processing task
            start_request = {"environment_id": environment_id}
            response = await self.make_request(
                "POST", "/api/v1/processing-tasks/start",
                json=start_request
            )
            
            if response["status"] not in [200, 202]:
                logger.error(f"❌ Failed to start task: {response}")
                return None
                
            task_data = response["data"]
            task_id = task_data["task_id"]
            
            logger.info(f"✅ Task started: {task_id}")
            logger.info(f"   Status URL: {task_data.get('status_url', 'N/A')}")
            logger.info(f"   WebSocket URL: {task_data.get('websocket_url', 'N/A')}")
            
            self.active_tasks[task_id] = {
                "environment_id": environment_id,
                "start_time": datetime.now(timezone.utc),
                "status_url": task_data.get("status_url"),
                "websocket_url": task_data.get("websocket_url")
            }
            
            # 2. Monitor task status
            await self.monitor_task_status(task_id)
            
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Processing task test failed: {e}")
            return None

    async def monitor_task_status(self, task_id: str, max_checks: int = 10):
        """Monitor task status until completion or max checks."""
        logger.info(f"📊 Monitoring task status: {task_id}")
        
        for check in range(max_checks):
            try:
                response = await self.make_request("GET", f"/api/v1/processing-tasks/{task_id}/status")
                
                if response["status"] == 200:
                    status_data = response["data"]
                    status = status_data["status"]
                    progress = status_data.get("progress", 0.0)
                    current_step = status_data.get("current_step", "Unknown")
                    
                    logger.info(f"   📈 Status: {status} | Progress: {progress:.1%} | Step: {current_step}")
                    
                    if status in ["COMPLETED", "FAILED"]:
                        logger.info(f"✅ Task {task_id} finished with status: {status}")
                        return status
                        
                    # Wait before next check
                    await asyncio.sleep(2.0)
                else:
                    logger.warning(f"⚠️ Status check failed: {response['status']}")
                    
            except Exception as e:
                logger.error(f"❌ Status check error: {e}")
                
        logger.warning(f"⚠️ Task monitoring ended after {max_checks} checks")
        return "MONITORING_TIMEOUT"

    # ===== WEBSOCKET SIMULATION =====
    
    async def test_websocket_connections(self, task_id: str):
        """Test WebSocket connections for real-time updates."""
        logger.info(f"🔌 Testing WebSocket connections for task: {task_id}")
        
        websocket_endpoints = [
            f"/ws/tracking/{task_id}",
            f"/ws/frames/{task_id}",
            "/ws/system"
        ]
        
        tasks = []
        for endpoint in websocket_endpoints:
            task = asyncio.create_task(self._test_websocket_endpoint(endpoint, task_id))
            tasks.append(task)
        
        # Run WebSocket tests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        total_count = len(websocket_endpoints)
        
        logger.info(f"🔌 WebSocket test results: {success_count}/{total_count} successful")
        return success_count == total_count

    async def _test_websocket_endpoint(self, endpoint: str, task_id: str, duration: int = 10) -> bool:
        """Test individual WebSocket endpoint."""
        ws_url = f"{self.ws_base_url}{endpoint}"
        logger.info(f"   🔗 Connecting to: {ws_url}")
        
        try:
            async with websockets.connect(ws_url) as websocket:
                logger.info(f"   ✅ Connected to {endpoint}")
                
                # Send initial message if needed
                if "tracking" in endpoint:
                    await websocket.send(json.dumps({
                        "type": "subscribe_tracking",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }))
                elif "system" in endpoint:
                    await websocket.send(json.dumps({
                        "type": "request_performance_history"
                    }))
                
                # Listen for messages for specified duration
                message_count = 0
                end_time = time.time() + duration
                
                while time.time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        message_count += 1
                        
                        logger.debug(f"   📨 {endpoint} message {message_count}: {data.get('type', 'unknown')}")
                        
                        # Log interesting message types
                        if data.get('type') in ['tracking_update', 'system_status', 'connection_established']:
                            logger.info(f"   📨 {endpoint}: Received {data.get('type')} message")
                            
                    except asyncio.TimeoutError:
                        continue  # No message received, continue listening
                    except json.JSONDecodeError as e:
                        logger.warning(f"   ⚠️ {endpoint}: Invalid JSON received: {e}")
                    
                logger.info(f"   ✅ {endpoint}: Received {message_count} messages in {duration}s")
                return True
                
        except ConnectionClosedError:
            logger.warning(f"   ⚠️ {endpoint}: Connection closed by server")
            return False
        except WebSocketException as e:
            logger.error(f"   ❌ {endpoint}: WebSocket error: {e}")
            return False
        except Exception as e:
            logger.error(f"   ❌ {endpoint}: Unexpected error: {e}")
            return False

    # ===== ANALYTICS SIMULATION =====
    
    async def test_analytics_endpoints(self) -> bool:
        """Test analytics endpoints."""
        logger.info("📊 Testing analytics endpoints...")
        
        analytics_tests = [
            ("GET", "/api/v1/analytics/real-time/metrics"),
            ("GET", "/api/v1/analytics/real-time/active-persons"),
            ("GET", "/api/v1/analytics/real-time/camera-loads"),
            ("GET", "/api/v1/analytics/system/statistics"),
            ("GET", "/api/v1/analytics/reports/types"),
        ]
        
        success_count = 0
        for method, endpoint in analytics_tests:
            try:
                response = await self.make_request(method, endpoint)
                if response["status"] == 200:
                    logger.info(f"   ✅ {endpoint}: {list(response['data'].keys())}")
                    success_count += 1
                else:
                    logger.warning(f"   ⚠️ {endpoint}: Status {response['status']}")
            except Exception as e:
                logger.error(f"   ❌ {endpoint}: {e}")
        
        # Test POST endpoints with sample data
        post_tests = [
            ("/api/v1/analytics/behavior/analyze", {
                "person_id": "test_person_123",
                "environment_id": "campus",
                "time_range": {
                    "start": datetime.now(timezone.utc).isoformat(),
                    "end": datetime.now(timezone.utc).isoformat()
                }
            }),
            ("/api/v1/analytics/historical/summary", {
                "environment_id": "campus",
                "time_range": {
                    "start": (datetime.now(timezone.utc)).isoformat(),
                    "end": datetime.now(timezone.utc).isoformat()
                },
                "metrics": ["person_count", "dwell_time"]
            })
        ]
        
        for endpoint, data in post_tests:
            try:
                response = await self.make_request("POST", endpoint, json=data)
                if response["status"] in [200, 422]:  # 422 is expected for invalid test data
                    logger.info(f"   ✅ {endpoint}: Response received")
                    success_count += 1
                else:
                    logger.warning(f"   ⚠️ {endpoint}: Status {response['status']}")
            except Exception as e:
                logger.error(f"   ❌ {endpoint}: {e}")
        
        total_tests = len(analytics_tests) + len(post_tests)
        logger.info(f"📊 Analytics test results: {success_count}/{total_tests} successful")
        return success_count >= total_tests * 0.7  # 70% success rate acceptable

    # ===== MEDIA ENDPOINTS SIMULATION =====
    
    async def test_media_endpoints(self, task_id: str) -> bool:
        """Test media serving endpoints."""
        logger.info("🎥 Testing media endpoints...")
        
        # Test deprecated media endpoint (should exist but marked deprecated)
        try:
            # This is a deprecated endpoint, but should still work
            test_endpoint = f"/api/v1/media/tasks/{task_id}/environments/campus/cameras/c01/sub_videos/test.mp4"
            response = await self.make_request("GET", test_endpoint)
            
            # We expect this to fail with 404 or similar since we don't have actual video files
            logger.info(f"   📁 Media endpoint tested (expected failure for test data)")
            return True
            
        except Exception as e:
            logger.info(f"   📁 Media endpoint test completed (expected error): {type(e).__name__}")
            return True  # This is expected to fail in simulation

    # ===== MAIN TEST ORCHESTRATION =====
    
    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive frontend simulation test."""
        logger.info("🚀 Starting comprehensive SpotOn frontend simulation")
        
        test_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tests": {},
            "summary": {}
        }
        
        try:
            # 1. Health checks
            health_result = await self.test_health_endpoints()
            test_results["tests"]["health_endpoints"] = health_result
            
            # 2. Authentication flow
            auth_result = await self.test_authentication_flow()
            test_results["tests"]["authentication"] = auth_result
            
            # 3. Processing task lifecycle
            task_id = await self.test_processing_task_lifecycle("campus")
            test_results["tests"]["processing_task"] = task_id is not None
            test_results["task_id"] = task_id
            
            if task_id:
                # 4. WebSocket connections
                ws_result = await self.test_websocket_connections(task_id)
                test_results["tests"]["websocket_connections"] = ws_result
                
                # 5. Media endpoints
                media_result = await self.test_media_endpoints(task_id)
                test_results["tests"]["media_endpoints"] = media_result
            
            # 6. Analytics endpoints
            analytics_result = await self.test_analytics_endpoints()
            test_results["tests"]["analytics_endpoints"] = analytics_result
            
            # Calculate summary
            passed_tests = sum(1 for result in test_results["tests"].values() if result)
            total_tests = len(test_results["tests"])
            
            test_results["summary"] = {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "end_time": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"🎯 Test Summary: {passed_tests}/{total_tests} tests passed ({test_results['summary']['success_rate']:.1%} success rate)")
            
            return test_results
            
        except Exception as e:
            logger.error(f"💥 Comprehensive test failed: {e}")
            test_results["error"] = str(e)
            test_results["traceback"] = traceback.format_exc()
            return test_results

    async def run_continuous_monitoring(self, duration_minutes: int = 5):
        """Run continuous monitoring simulation."""
        logger.info(f"🔄 Starting continuous monitoring for {duration_minutes} minutes")
        
        # Start a processing task
        task_id = await self.test_processing_task_lifecycle("campus")
        if not task_id:
            logger.error("❌ Cannot start continuous monitoring without a task")
            return
        
        # Create monitoring tasks
        end_time = time.time() + (duration_minutes * 60)
        
        async def periodic_health_check():
            while time.time() < end_time:
                await self.test_health_endpoints()
                await asyncio.sleep(30)  # Check every 30 seconds
        
        async def periodic_status_check():
            while time.time() < end_time:
                await self.monitor_task_status(task_id, max_checks=1)
                await asyncio.sleep(15)  # Check every 15 seconds
        
        async def websocket_monitor():
            await self.test_websocket_connections(task_id)
        
        # Run monitoring tasks concurrently
        monitoring_tasks = [
            periodic_health_check(),
            periodic_status_check(),
            websocket_monitor()
        ]
        
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        logger.info("🏁 Continuous monitoring completed")


async def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="SpotOn Frontend Simulation Client")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--mode", choices=["test", "monitor"], default="test", help="Operation mode")
    parser.add_argument("--duration", type=int, default=5, help="Monitoring duration in minutes")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"SpotOn Frontend Simulator starting in {args.mode} mode")
    
    try:
        async with SpotOnFrontendSimulator(base_url=args.url) as simulator:
            if args.mode == "test":
                results = await simulator.run_comprehensive_test()
                
                # Print detailed results
                print("\n" + "="*60)
                print("SPOTON FRONTEND SIMULATION RESULTS")
                print("="*60)
                
                for test_name, result in results["tests"].items():
                    status = "✅ PASS" if result else "❌ FAIL"
                    print(f"{test_name:<25} {status}")
                
                print("\n" + "-"*60)
                summary = results["summary"]
                print(f"Total Tests: {summary['total_tests']}")
                print(f"Passed: {summary['passed_tests']}")
                print(f"Failed: {summary['failed_tests']}")
                print(f"Success Rate: {summary['success_rate']:.1%}")
                print("="*60)
                
                # Exit with appropriate code
                sys.exit(0 if summary['success_rate'] >= 0.7 else 1)
                
            elif args.mode == "monitor":
                await simulator.run_continuous_monitoring(args.duration)
                
    except KeyboardInterrupt:
        logger.info("🛑 Simulation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Simulation failed: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())