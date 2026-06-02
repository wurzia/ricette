#!/bin/sh
# Usage: ./deploy.sh
# Creates the persistent data volume if it doesn't already exist, then deploys.
set -e

APP="ricette"
VOLUME="ricette_data"
REGION="iad"

if ! fly volumes list -a "$APP" | grep -q "$VOLUME"; then
  echo "Creating volume $VOLUME in $REGION..."
  fly volumes create "$VOLUME" --app "$APP" --region "$REGION" --size 1
else
  echo "Volume $VOLUME already exists, skipping."
fi

fly deploy
