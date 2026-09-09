# Joint Ramp Metering + Variable Speed Limiting (RM + VSL) — Research Variant Documentation

## 1. Concept & Contribution

while the ramp metering alone can control the on-ramp inflow to reduce potential traffic congestion at the merge zone (the bottleneck), the mainline inflow still has big influence on the traffic behaviour at the merge zone that the ramp meter can not reach to handle. our idea brings a second, complementary control mechanism to the mainline axis: a vsl controller placed upstream the merge zone within a reasonably long ground. the vsl placement and concept is motivated and supported by the "SPECIALIST" paper's theory and intuition, which demonstrates analytically that upstream VSL application is the necessary and sufficient condition for shockwave dissolution. However, unlike SPECIALIST, which reactively computes a single analytical VSL command upon jam detection, our approach learns a RL policy that proactively shapes the bottleneck's receiving conditions before congestion onset, using a fixed upstream control zone and a discrete speed action space.

Variant location: `env/custom_env/rm_vsl_macro_with_variables_vsl/`


---

## 3. VSL Zone: Location & Length Design

### 3.1 Positioning Decision

The VSL zone must start before the merging area and after the upstream induction loop detectors to keep the detectors in the uncontrolled `main_road` segment, just upstream of `vsl_begin` junction.

- Placing the detectors outside (upstream of) the VSL zone ensures they measure the **honest, uncontrolled incoming traffic state** - not the already-decelerated traffic resulting from the VSL action.

### 3.2 Length Decision

The original VSL zone was only 61.79m — approximately 2–3 seconds of travel time at highway speed, too short for the mechanism to work effectively. It has been enlarged to **~165m**.

**Travel time through the zone by VSL level:**
| VSL Level | Speed | Travel Time |
|---|---|---|
| 100 km/h (free-flow) | 27.78 m/s | ~5.9s |
| 75 km/h | 20.83 m/s | ~7.9s |
| 50 km/h | 13.89 m/s | ~11.8s |

**Justifications:**

1. **Realism & Deployability:** A 165m VSL zone with 40s update intervals is a practical, realistic setup comparable to on-ramp advisory speed sign installations on European motorways. It is short enough to be a single controlled segment, yet long enough to produce measurable traffic effects.

2. **Speed Harmonization Before Merge:** Slowing mainline vehicles in the VSL zone reduces the speed differential between them and merging on-ramp vehicles (~50 km/h). Eliminating this speed gap directly reduces forced braking at the bottleneck and the associated capacity drop.

3. **Natural Gap Creation:** When a vehicle platoon enters the VSL zone and decelerates, and then re-accelerates, past vsl_zone, the **time-headways between vehicles increase naturally**. This passively creates merge opportunities without the agent needing to model individual vehicle trajectories.

4. **Lane-Change Facilitation:** At 75 km/h over 165m, a driver has ~8 seconds inside the zone — sufficient time to identify a gap in the adjacent lane and execute a lane change, gradually clearing `vsl_zone_0` for on-ramp vehicles.

5. **Shock Wave Absorption (Upstream Prevention):** By creating a controlled, gentle deceleration _upstream_ of the bottleneck, the VSL zone prevents stop-and-go shockwaves from forming and propagating far upstream. This is consistent with the **SPECIALIST** algorithm (Hegyi et al., 2008) — a VSL control strategy grounded in kinematic wave theory — which shows that applying reduced speed limits at an appropriate distance upstream of a jam causes it to "starve" (outflow > inflow) and dissolve, while placing the VSL at or downstream of the jam is ineffective. *(Reference: Hegyi, A., Hoogendoorn, S. P., Schreuder, M., Stoelhorst, H., & Viti, F. (2008). SPECIALIST: A dynamic speed limit control algorithm based on shock wave theory. Proceedings of the 11th IEEE ITSC, pp. 827–832.)*

---

## 5. MDP Formulation (To Be Completed)

### 5.1 Action Space

21 discrete joint actions = 7 RM green times × 3 VSL levels.

- **Green time:** {10, 15, 20, 25, 30, 35, 40} seconds.
- **VSL speed:** {50, 75, 100} km/h applied to `vsl_zone_0` for the **full 40s cycle** (not only the green sub-phase).
- Decoding: `green_idx = action % 7`, `vsl_idx = action // 7`.

### 5.2 Observation Space

15-d normalized flat vector (unchanged from previous variant, with two lane-index bugs corrected — see Section 6):

| # | Feature | Source |
|---|---|---|
| 1–2 | Upstream mainline flow, merge area flow | `upstream_detector_ids`, `bottleneck_detector_ids` |
| 3–4 | Upstream occ, upstream speed | `upstream_detector_ids` |
| 5–6 | Bottleneck occ, bottleneck speed | `bottleneck_detector_ids` |
| 7 | Ramp queue (veh) | edge vehicle count on `on_ramp` |
| 8–10 | Lane-0 (VSL lane) flow, occ, speed at bottleneck | `bottleneck_detector_ids[1]` |
| 11–13 | Lane-0 (VSL lane) flow, occ, speed upstream | `upstream_detector_ids[0]` |
| 14–15 | Last chosen green time, last chosen VSL speed | previous action |

### 5.3 Reward Function

Weighted linear combination. Theoretical range: **≈ [−7, +4]**.

| Component | Type | Weight | Notes |
|---|---|---|---|
| Merge area speed | + | 1.5 | |
| Upstream speed | + | 1.0 | |
| Downstream speed | + | 0.5 | |
| Downstream throughput (outflow veh/h) | + | 1.0 | **New** — prevents "empty road" degenerate policy |
| Bottleneck occupancy | − | 2.0 | |
| Upstream occupancy | − | 1.0 | |
| Ramp queue (quadratic) | − | 3.0 | **Changed** — replaces linear queue (w=1) + spillback cliff (w=20) |

**Key reward changes from previous variant:**
1. `_penalty_ramp_queue` changed from linear (`-norm_q`) to **quadratic** (`-norm_q²`) — smooth escalating urgency instead of abrupt cliff at 90%.
2. `_penalty_spillback` **removed** — the cliff at 90% caused gradient instability; the quadratic penalty handles the full range continuously.
3. `_reward_throughput` **added** to `_calculate_reward` (it existed as a method but was unused) — balances speed rewards and discourages starving the ramp.

### 5.4 Network Architecture

Unchanged from previous variant — **Dueling Double DQN** with a 2-layer MLP body:

```
Input(15) → FC(256) → ReLU → FC(128) → ReLU → body_out(128)
                                                      ↓
                        Value head: FC(1)   +   Advantage head: FC(output_dim)
```

- `output_dim` is dynamically set to `action_space_n` (21) at construction — no manual change needed.
- The body (256 → 128) is kept for a fair comparison with the previous 42-action variant.
- A leaner body (128 → 64) is a candidate for future experimentation if convergence is slow.

---

## 6. Data Collection Bug Fixes (inherited from previous variant)

Two lane-index errors were identified and corrected in `rl_controller.py` when porting from the previous variant:

**Bug 1 — Upstream lane-0 detector (wrong lane):**
The three upstream lane-0 queries (`flow`, `occ`, `speed`) were reading from `upstream_detector_ids[1]` (`up_stream_sens_1`, on `main_road_1` — the middle lane) instead of `upstream_detector_ids[0]` (`up_stream_sens_0`, on `main_road_0` — the VSL lane). Fixed to `[0]`.

**Bug 2 — Bottleneck lane-0 detector (wrong lane):**
The three bottleneck lane-0 queries were reading from `bottleneck_detector_ids[0]` (`bottle_neck_sens_0`, on `acceleration_area_0` — the **ramp auxiliary merge lane**) instead of `bottleneck_detector_ids[1]` (`bottle_neck_sens_1`, on `acceleration_area_1` — the downstream continuation of `vsl_zone_0`). The lane numbering shifts by one at the merge junction because the ramp occupies lane 0 of `acceleration_area`. Fixed to `[1]`.
