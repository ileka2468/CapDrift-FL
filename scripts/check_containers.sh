#!/bin/bash
NODE=$(hostname)
echo "=== Checking Containers on $NODE ==="
for id in $(pct list | awk 'NR>1 {print $1}'); do
    # Skip templates (204, 212, 222)
    if [ "$id" = "204" ] || [ "$id" = "212" ] || [ "$id" = "222" ]; then
        continue
    fi
    VER=$(pct exec $id -- /opt/fl_client/venv/bin/python3 -c "import torch; print(torch.__version__)" 2>&1 || echo "ERR")
    IP=$(pct exec $id -- ip -4 -br a show eth0 2>&1 | awk '{print $3}' || echo "No IP")
    echo "  [OK] CT $id on $NODE ($IP) -> PyTorch: $VER"
done
