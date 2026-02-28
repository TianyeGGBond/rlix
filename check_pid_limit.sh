#!/bin/bash
# Deep-Dive PID/Thread Limit Checker - Verbose Edition

echo "======= LAYER 1: KERNEL-WIDE HARD LIMITS (HOST) ======="
# These values are the absolute ceiling for the entire machine.
if [ -f /proc/sys/kernel/pid_max ]; then
    echo "PATH: /proc/sys/kernel/pid_max"
    echo "DEFINITION: Global maximum PIDs allowed by the host kernel."
    echo "VALUE: $(cat /proc/sys/kernel/pid_max)"
fi
echo ""
if [ -f /proc/sys/kernel/threads-max ]; then
    echo "PATH: /proc/sys/kernel/threads-max"
    echo "DEFINITION: Global maximum threads allowed across all processes."
    echo "VALUE: $(cat /proc/sys/kernel/threads-max)"
fi



echo -e "\n======= LAYER 2: CONTAINER CGROUP LIMITS (CGROUP V2) ======="
if [ -f /sys/fs/cgroup/pids.max ]; then
    LIMIT=$(cat /sys/fs/cgroup/pids.max)
    USAGE=$(cat /sys/fs/cgroup/pids.current)
    
    echo "PATH:       /sys/fs/cgroup/pids.max"
    echo "VERSION:    Cgroup v2 (Unified Hierarchy)"
    echo "LIMIT:      $LIMIT (Maximum allowed PIDs/threads)"
    echo "USAGE:      $USAGE (Current active PIDs/threads)"
    
    # Calculate headroom if LIMIT is a number
    if [[ "$LIMIT" =~ ^[0-9]+$ ]]; then
        FREE=$((LIMIT - USAGE))
        echo "HEADROOM:   $FREE (Remaining PIDs before fork failure)"
    else
        echo "HEADROOM:   Unlimited (Limit is set to 'max')"
    fi
fi


echo -e "\n======= LAYER 3: USER-LEVEL CONSTRAINTS (ULIMIT) ======="
echo "PATH: /proc/self/limits (Max processes)"
echo "DEFINITION: Per-UID limit. If multiple containers share a UID, they share this limit."
grep "Max processes" /proc/self/limits

echo -e "\n======= LAYER 4: MEMORY-BASED THREAD LIMIT (STACK) ======="
echo "COMMAND: ulimit -s"
echo "DEFINITION: Stack size per thread. High values crash the container via OOM before hitting PID limits."
echo "VALUE: $(ulimit -s) KB"

echo -e "\n======= LAYER 5: CGROUP THROTTLE EVENTS ======="
if [ -f /sys/fs/cgroup/pids.events ]; then
    echo "PATH: /sys/fs/cgroup/pids.events"
    echo "DEFINITION: Count of how many times a 'fork' was blocked because of the limit."
    cat /sys/fs/cgroup/pids.events
else
    # Legacy v1 fallback
    echo "STATUS: pids.events not found (likely cgroup v1 or no limit set)."
fi

echo -e "\n======= LAYER 6: CURRENT RESOURCE CONSUMPTION ======="
echo "DEFINITION: Real-time count of active Threads vs Processes."
echo "Total Processes (PIDs): $(ps -e --no-headers | wc -l)"
echo "Total Threads (LWPs):   $(ps -eL --no-headers | wc -l)"

