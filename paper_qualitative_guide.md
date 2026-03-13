# Qualitative UI Instructions

To capture real qualitative visual results for Figures 4 and 6, follow these exact steps:

1. START THE SYSTEM (Ensure Docker/Database is running)
   Run: docker compose -f docker-compose.gpu.yml up -d

2. LAUNCH THE FRONTEND DASHBOARD
   Navigate to spoton_frontend and run: npm run dev

3. CAPTURE FIGURE 6 (Floorplan Dashboard)
   - Open your browser to http://localhost:3000
   - Go to the "Analytics" or "Dashboard" view showing your map.
   - Inject heavy traffic (wait for people to appear if using a live file).
   - Take a large, clean screenshot.

4. CAPTURE FIGURE 4 (Re-ID Track Handoffs)
   - Keep the dashboard open, watching the camera popups.
   - Wait for a person to walk out of one camera frame, and into the next.
   - Look for the colored bounding box/trajectory line with the SAME User ID.
   - Take cropped screenshots of the two camera streams showing the matched ID.
    
    