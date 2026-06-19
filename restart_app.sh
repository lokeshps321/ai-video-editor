#!/bin/bash
# Quick restart script for ClipMind

echo "🔄 Restarting ClipMind services..."

# Kill existing processes
echo "Stopping old processes..."
pkill -f "uvicorn app.main" 2>/dev/null
pkill -f "node.*vite" 2>/dev/null
pkill -f "bin/vite" 2>/dev/null
sleep 2

# Clear memory cache (optional, helps with OOM)
echo "Clearing memory cache..."
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null || true

# Start backend
echo "Starting backend..."
cd /home/lokesh/ai_video_editor/backend
unset DATABASE_URL
nohup python3 -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd /home/lokesh/ai_video_editor/frontend
nohup npx vite --host 0.0.0.0 > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

# Wait and verify
sleep 6

echo ""
echo "=== Status ==="
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend: Running (port 8000)"
else
    echo "❌ Backend: Failed to start"
fi

if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo "✅ Frontend: Running (port 5173)"
else
    echo "❌ Frontend: Failed to start"
fi

echo ""
echo "📍 Open: http://localhost:5173"
echo ""
echo "Logs:"
echo "  Backend:  /tmp/backend.log"
echo "  Frontend: /tmp/frontend.log"
