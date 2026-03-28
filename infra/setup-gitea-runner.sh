#!/bin/bash
set -e

GITEA_URL="http://gitea.bb.rctechpr.net"
GITEA_USER="localadmin"
GITEA_PASS="localadmin"
GITHUB_REPO="https://github.com/nalditopr/ollama.git"
GITEA_REPO="ollama-turboquant"

echo "=== Step 1: Create Gitea repo (mirror from GitHub) ==="
curl -s -X POST "$GITEA_URL/api/v1/repos/migrate" \
  -u "$GITEA_USER:$GITEA_PASS" \
  -H "Content-Type: application/json" \
  -d "{
    \"clone_addr\": \"$GITHUB_REPO\",
    \"repo_name\": \"$GITEA_REPO\",
    \"repo_owner\": \"$GITEA_USER\",
    \"mirror\": true,
    \"service\": \"github\"
  }" | jq .

echo "=== Step 2: Get runner registration token ==="
TOKEN=$(curl -s -X GET "$GITEA_URL/api/v1/repos/$GITEA_USER/$GITEA_REPO/actions/runners/registration-token" \
  -u "$GITEA_USER:$GITEA_PASS" | jq -r .token)
echo "Runner token: $TOKEN"
echo "Use this token in gitea-runner.tf or pass to act_runner register"

echo "=== Step 3: Enable Actions in Gitea ==="
echo "Add to app.ini [actions] section:"
echo "  ENABLED = true"
echo "  DEFAULT_ACTIONS_URL = https://github.com"
echo ""
echo "Then restart Gitea pod."

echo "=== Step 4: Deploy runner ==="
echo "Run: tofu apply -var gitea_runner_token=$TOKEN"
