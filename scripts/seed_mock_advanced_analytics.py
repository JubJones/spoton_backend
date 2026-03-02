import asyncio
import logging
from datetime import datetime, timedelta, timezone
import random
import sys
import os
import math

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.infrastructure.database.base import get_session_factory, setup_database
from app.infrastructure.database.models.tracking_models import TrackingEvent, PersonTrajectory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_advanced_analytics")

def interpolate(p1, p2, fraction):
    return (p1[0] + (p2[0] - p1[0]) * fraction, p1[1] + (p2[1] - p1[1]) * fraction)

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def generate_mock_data():
    session_factory = get_session_factory()
    if not session_factory:
        logger.error("No session factory")
        return
        
    session = session_factory()
    try:
        # First, purge old mock data to prevent massive inflation
        logger.info("Purging old mock data...")
        session.query(TrackingEvent).filter(TrackingEvent.environment_id.in_(["factory", "campus"])).delete()
        session.query(PersonTrajectory).filter(PersonTrajectory.environment_id.in_(["factory", "campus"])).delete()
        session.commit()

        now = datetime.now(timezone.utc)
        
        for env_id, env_cfg in settings.ENVIRONMENT_TEMPLATES.items():
            cameras = list(env_cfg.get("cameras", {}).keys())
            if not cameras:
                continue
            
            logger.info(f"Generating realistic mock data for {env_id}")
            
            tracking_events = []
            trajectories = []
            
            # Define specific routes through the environment (x, y, camera_id)
            if env_id == 'factory':
                routes = [
                    # Normal worker shifts (Entrance -> PL1 -> QC -> Exit)
                    [(2, 5, 'c13'), (15, 5, 'c13'), (35, 5, 'c16'), (55, 5, 'c12'), (70, 5, 'c09')],
                    # Manager walk around
                    [(70, 8, 'c09'), (55, 8, 'c12'), (35, 8, 'c16'), (15, 8, 'c13'), (2, 8, 'c13')],
                    # Quick exit
                    [(35, 2, 'c16'), (15, 2, 'c13'), (5, 2, 'c13')]
                ]
            else:
                routes = [
                    # Student path (Entrance -> Plaza -> Walkway -> Commons)
                    [(5, 2, 'c01'), (30, 6, 'c02'), (50, 7, 'c03'), (68, 4, 'c05')],
                    # Reverse student path
                    [(68, 4, 'c05'), (50, 7, 'c03'), (30, 6, 'c02'), (5, 2, 'c01')],
                    # Short hangouts
                    [(30, 6, 'c02'), (32, 8, 'c02'), (30, 10, 'c02')]
                ]
            
            # Generate past 24 hours of data
            for hours_back in range(24):
                base_time = now - timedelta(hours=hours_back)
                
                # Active hours have more traffic
                is_active_hour = 8 <= base_time.hour <= 18
                num_personas = random.randint(30, 80) if is_active_hour else random.randint(5, 15)
                
                for t in range(num_personas):
                    person_id = f"real-mock-{env_id}-{hours_back}-{t}"
                    route = random.choice(routes)
                    
                    # Persona speed (px/s or meters/s)
                    speed = random.uniform(1.0, 2.5)
                    
                    current_time = base_time + timedelta(seconds=random.randint(0, 3500))
                    
                    seq_num = 0
                    for i in range(len(route) - 1):
                        start_wp = route[i]
                        end_wp = route[i+1]
                        
                        dist = distance((start_wp[0], start_wp[1]), (end_wp[0], end_wp[1]))
                        if dist == 0: continue
                        
                        time_to_travel = dist / speed
                        # Generate points every ~1 second
                        num_points = int(time_to_travel)
                        if num_points == 0: num_points = 1
                        
                        # Add some jitter
                        jitter_x = random.uniform(-0.5, 0.5)
                        jitter_y = random.uniform(-0.5, 0.5)

                        vx = (end_wp[0] - start_wp[0]) / time_to_travel
                        vy = (end_wp[1] - start_wp[1]) / time_to_travel
                        
                        for p in range(num_points):
                            fraction = p / num_points
                            curr_x, curr_y = interpolate((start_wp[0], start_wp[1]), (end_wp[0], end_wp[1]), fraction)
                            
                            curr_x += jitter_x
                            curr_y += jitter_y
                            
                            point_time = current_time + timedelta(seconds=p)
                            cam = start_wp[2] if fraction < 0.5 else end_wp[2] # Simulate handoff mid-way
                            
                            tracking_events.append(
                                TrackingEvent(
                                    timestamp=point_time,
                                    global_person_id=person_id,
                                    camera_id=cam,
                                    environment_id=env_id,
                                    position_x=curr_x,
                                    position_y=curr_y,
                                    event_type="tracking"
                                )
                            )
                            
                            trajectories.append(
                                PersonTrajectory(
                                    timestamp=point_time,
                                    global_person_id=person_id,
                                    camera_id=cam,
                                    environment_id=env_id,
                                    sequence_number=seq_num,
                                    position_x=curr_x,
                                    position_y=curr_y,
                                    velocity_x=vx,
                                    velocity_y=vy
                                )
                            )
                            seq_num += 1
                        
                        current_time += timedelta(seconds=num_points)
            
            # Batch insert
            num_events = len(tracking_events)
            num_traj = len(trajectories)
            logger.info(f"Inserting {num_events} tracking events and {num_traj} trajectories for {env_id}")
            
            chunk_size = 5000
            for i in range(0, len(tracking_events), chunk_size):
                session.bulk_save_objects(tracking_events[i:i+chunk_size])
                session.commit()
                
            for i in range(0, len(trajectories), chunk_size):
                session.bulk_save_objects(trajectories[i:i+chunk_size])
                session.commit()
                
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to seed advanced analytics data: {e}")
    finally:
        session.close()

async def main():
    await setup_database()
    generate_mock_data()

if __name__ == "__main__":
    asyncio.run(main())
