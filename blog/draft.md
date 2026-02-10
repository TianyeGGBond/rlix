challenge
- manage version of cached checkpoint e.g. one-step-off, update interval, rollpacker like on-policy but overlaps with multiple steps 
- inflight model update during rollout/generation 

| Dimension | Synchronous Co-location | Asynchronous Separation | Time-Division Multiplexing |
|---|---|---|---|
| Parallelism | ❌ No parallelism | ⚠️ Fixed pipeline | ✅ Dynamic parallelism |
| Resource Waste | ❌ Severe | ⚠️ Medium (bubbles on both sides) | ✅ Minimal |
| Implementation Complexity | ✅ Low | ✅ Medium | ⚠️ High |