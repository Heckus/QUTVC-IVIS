#!/bin/bash

echo "🚀 Starting YOLOv8 Dataset Manager..."

# --- Activate Environments ---
echo "🐍 Activating Python virtual environment..."
source ../venv/bin/activate

echo "📦 Activating Node.js environment..."
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

# --- Start Backend Server ---
echo "🖥️  Starting backend server on port 3000..."
node server.js &
SERVER_PID=$!

# Add a small delay to give the server time to start
sleep 2

# --- Start Frontend Server ---
PORT=8000
echo "----------------------------------------------------"
echo ""
echo "  App is running! Open the link below in your browser."
echo "  👉 http://localhost:$PORT"
echo ""
echo "----------------------------------------------------"
echo "(Press Ctrl+C to stop everything)"

# Define a cleanup function to run when you press Ctrl+C
cleanup() {
    echo -e "\n🛑 Stopping backend server (PID: $SERVER_PID)..."
    kill $SERVER_PID
    exit
}

# Trap Ctrl+C and call the cleanup function
trap cleanup INT

# Serve the frontend and wait
python3 -m http.server $PORT > /dev/null 2>&1 &
wait $SERVER_PID