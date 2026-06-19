#!/bin/bash

# Cleanup script for AI Video Editor
# Deletes files older than 24 hours in uploads, renders, and tmp directories

UPLOADS_DIR="/home/lokesh/ai video editor/backend/uploads"
RENDERS_DIR="/home/lokesh/ai video editor/backend/renders"
TMP_DIR="/home/lokesh/ai video editor/backend/tmp"

# Find and delete files older than 1 day (-mtime +0 usually means > 24 hours, but -mtime +1 means strictly > 48 hours for some find implementations. Wait, let's use -mmin +1440 for exactly 24 hours to be safe)

echo "Starting cleanup at $(date)"

# Delete files older than 24 hours (1440 minutes)
find "$UPLOADS_DIR" -type f -mmin +1440 -exec rm -f {} \;
find "$RENDERS_DIR" -type f -mmin +1440 -exec rm -f {} \;
find "$TMP_DIR" -type f -mmin +1440 -exec rm -f {} \;

echo "Cleanup finished at $(date)"
