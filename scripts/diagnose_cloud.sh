#!/bin/bash
echo "=== Who am I ==="
whoami
id
echo "=== Working directory ==="
pwd
ls -la
echo "=== Models directory permissions ==="
ls -lad models
ls -la models
echo "=== Test writing to models/ ==="
touch models/test_write.txt && echo "Write to models/ SUCCESS" || echo "Write to models/ FAILED"
echo "=== Test writing to /tmp/ ==="
touch /tmp/test_write.txt && echo "Write to /tmp/ SUCCESS" || echo "Write to /tmp/ FAILED"
echo "=== Test chmod 777 models ==="
chmod 777 models || sudo chmod 777 models || true
touch models/test_write2.txt && echo "Write after chmod SUCCESS" || echo "Write after chmod FAILED"
