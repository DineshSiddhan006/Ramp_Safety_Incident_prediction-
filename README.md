
# GH-003 Ramp Safety Incident Prediction Engine Dataset

Synthetic dataset representing 8-hour ground handling shifts at JED airport.
Total records: 40,000

Files:
- ramp_safety_train.csv (28,000)
- ramp_safety_validation.csv (6,000)
- ramp_safety_test.csv (6,000)

Target:
- safety_risk_score (0.00-1.00)
- risk_category (LOW_RISK, MODERATE_RISK, HIGH_RISK)

Distribution:
- LOW_RISK: 51%
- MODERATE_RISK: 30%
- HIGH_RISK: 19%

## 1. Project Charter & Executive Overview
* **Prediction ID:** GH-003
* **Operational Domain:** Aviation Ground Handling & Apron Tarmac Operations
* **Station Context:** King Abdulaziz International Airport (JED) — Jeddah, Saudi Arabia
* **Core Objective:** Predict the continuous safety risk volatility score and categorical classification per 8-hour operational shift window to drive proactive safety management interventions.
* **Key Personas Impacted:** VP of Safety, CEO, Ground Operations Director
* **Business Value Realized:** Prevention of multi-million dollar aircraft hull damage (e.g., wingtip clips, fuselage skin punctures), reduction of airside personnel injuries, decrease in aviation insurance premiums, and complete compliance with GACA (General Authority of Civil Aviation) standards.

---

## 2. Dataset Partitioning & Target Mandate
The dataset comprises a total of **40,000 unique operational shift records** spanning from March 2025 to February 2026. The dataset has been thoroughly shuffled across its temporal timeline to eliminate seasonal data leakage, and separated into a strict **70% / 15% / 15%** configuration:

* **`ramp_safety_train.csv`:** 28,000 rows
* **`ramp_safety_validation.csv`:** 6,000 rows
* **`ramp_safety_test.csv`:** 6,000 rows

### Target Variables & Distribution Proportions
The framework tracks a dual target layout matching these exact dataset proportions:
1. **`safety_risk_score`:** Continuous Float value tracking underlying shift volatility. Bound strictly between `0.00` (Nominal safety state) and `1.00` (Imminent incident threshold).
2. **`risk_category`:** Categorical String derived natively from the risk score boundaries:
    * **`LOW_RISK` (Exactly 51%):** Standard operations operating smoothly within safe baseline parameters (`safety_risk_score` < 0.50).
    * **`MODERATE_RISK` (Exactly 30%):** Operations facing minor equipment faults, weekend traffic pressure, or moderate congestion (`0.50` <= `safety_risk_score` < `0.75`).
    * **`HIGH_RISK` (Exactly 19%):** Volatile operations experiencing dangerous combinations of extreme heat, severe wind fronts, or high fatigue (`safety_risk_score` >= 0.75).
---
## 3. Comprehensive Feature Dictionary & Logical Boundaries
### Operational Identification Features
* **`shift_id`:** Alphanumeric unique key matching format `JED-SHIFT-XXXXX`. Used as a primary row identifier; excluded from active machine learning model training.
* **`timestamp`:** ISO 8601 UTC string format (`YYYY-MM-DDTHH:MM:SSZ`) identifying the exact 8-hour operational shift start period.
### Environmental Pillar
* **`weather_condition` [Categorical String]**
  * *Description:* The primary prevailing atmospheric state during the shift block.
  * *Logical Bounds:* N/A
  * *Categorical Value Meanings:*
    * `CLEAR`: Clear skies, nominal conditions, standard visibility.
    * `EXTREME_HEAT`: Severe temperature spikes altering equipment performance limits.
    * `DUST_HAZE`: Suspended particulate matter reducing mid-range spatial visibility.
    * `SANDSTORM`: Active desert convective wall causing immediate line-of-sight failure.
* **`visibility_meters` [Integer]**
  * *Description:* The maximum horizontal distance at which aircraft markings and tarmac boundaries can be clearly seen by airside personnel.
  * *Logical Bounds:* Min: `200` meters | Max: `9,999` meters.
* **`temperature_celsius` [Continuous Float]**
  * *Description:* The measured ambient outdoor air temperature at the open aircraft apron.
  * *Logical Bounds:* Min: `15.0`°C | Max: `49.0°C` (reflecting extreme summer heat radiating off asphalt heat-sinks).
* **`wind_speed_kmph` [Continuous Float]**
  * *Description:* The sustained wind velocity across the unprotected apron environment.
  * *Logical Bounds:* Min: `0.0` km/h | Max: `65.0` km/h (The absolute regulatory ceiling; at 65+ km/h all ramp actions are legally shut down by GACA).
### Human Factors Pillar
* **`worker_fatigue_hours` [Continuous Float]**
  * *Description:* The average cumulative number of consecutive duty hours logged by the shift crew on heavy physical ramp assignments without a mandatory rest window.
  * *Logical Bounds:* Min: `0.0` hours | Max: `12.0` hours (The maximum statutory emergency extension cap).
* **`overtime_workers_count` [Integer]**
  * *Description:* The exact headcount of ground handling personnel on the current shift who are extending past their standard rostered weekly profiles.
  * *Logical Bounds:* Min: `0` workers | Max: `20` workers (capped at 20 to preserve realistic shift-level team boundaries).
### Equipment State Pillar
* **`equipment_fault_count` [Integer]**
  * *Description:* The active tally of mechanical, hydraulic, or electrical maintenance anomalies logged across Ground Support Equipment (GSE) fleets during the shift.
  * *Logical Bounds:* Min: `0` faults | Max: `10` faults.
* **`communication_failure_count` [Integer]**
  * *Description:* The number of wireless radio dropouts, telemetry lags, or dead aircraft interphone headset connections documented during operations.
  * *Logical Bounds:* Min: `0` failures | Max: `6` failures.
### Workload Intensity & Temporal Pillar
* **`active_staff_count` [Integer]**
  * *Description:* The total physical workforce (loaders, drivers, supervisors) active on the tarmac grid during the shift.
  * *Logical Bounds:* Min: `40` workers | Max: `250` workers.
* **`aircraft_on_ramp_count` [Integer]**
  * *Description:* The simultaneous number of aircraft parked on terminal gates or remote stands requiring active turnaround services.
  * *Logical Bounds:* Min: `1` aircraft | Max: `25` aircraft (constrained by physical sector spatial allocations).
* **`day_traffic_profile` [Categorical String]**
  * *Description:* Chronological category derived directly from the timestamp calendar day.
  * *Logical Bounds:* N/A
  * *Categorical Value Meanings:*
    * `WEEKDAY_CALM`: Sunday through Thursday; routine scheduled business and cargo movements.
    * `WEEKEND_RUSH`: Friday and Saturday; major peak travel volume spikes matching regional administrative weekend layouts.

---

## 4. Feature Predictive Contributions (100% Impact Weight Allocation)

The table below describes how predictive weight is distributed across features under a tree-based model framework (**XGBoost/CatBoost**), indicating whether they impact safety baselines **individually (Main Effect)** or only reveal severe hazards **in combinations (Interaction Effects)**.

| Feature Name | Impact Tier | Weight % | Mode of Action | Operational Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`visibility_meters`** | High | **16%** | Individual + Combination | Drops linearly increase baseline risk due to loss of line-of-sight. Spikes exponentially into high risk when paired with high `active_vehicle_count`. |
| **`wind_speed_kmph`** | High | **14%** | Individual + Combination | High speeds individually increase risk of loose debris projectiles (FOD). Escalates when combined with complex night shifts. |
| **`worker_fatigue_hours`** | High | **13%** | Individual + Combination | Steady main effect as human reflexes slow over time. Triggers extreme high-risk spikes when combined with intense afternoon heat. |
| **`overtime_workers_count`**| High | **12%** | Individual + Combination | Proxy for team-wide scheduling fatigue and shortcut behavior. Combines with fatigue hours to multiply systemic human error. |
| **`equipment_fault_count`** | High | **11%** | Individual + Combination | Main effect from mechanical unpredictability. Combines with time compression to force manual workarounds. |
| **`aircraft_on_ramp_count`**| High | **10%** | Individual + Combination | Structural indicator for apron spatial congestion. Drives up the interaction risk when paired with the `WEEKEND_RUSH` flag. |
| **`temperature_celsius`** | Moderate| **8%** | Combination Dominant | Minimal standalone impact on the machinery itself, but acts as a massive risk multiplier when paired with high human fatigue. |
| **`communication_failure_count`**| Moderate| **7%** | Combination Dominant | Low standalone risk, but highly volatile when combined with heavy congestion, causing operators to maneuver heavy equipment blind. |
| **`weather_condition`** | Moderate| **5%** | Individual | Sets the underlying baseline environmental rules for speed limits and crew safety configurations. |
| **`active_staff_count`** | Low | **3%** | Individual | Tracks general airside visual noise and crowd density. |
| **`day_traffic_profile`** | Low | **1%** | Combination Dominant | Zero direct isolated impact. Serves strictly as a background catalyst that alerts the engine that high workload features are peaking. |
| **TOTAL** | | **100%** | | |

---

## 5. Deterministic Operational Causal Logic
To ensure robust, non-linear machine learning modeling, the dataset bypasses random stochastic noise in favor of deterministic interaction filters:

1. **The Coastal Sandstorm Interaction (`HIGH_RISK`):** `weather_condition == "SANDSTORM"` + `visibility_meters < 500` + `wind_speed_kmph > 45.0` + `aircraft_on_ramp_count > 15` $\rightarrow$ Forces `safety_risk_score` >= `0.80`.
2. **The Thermal Fatigue Overload (`HIGH_RISK`):** `temperature_celsius > 44.0`°C + `worker_fatigue_hours > 8.0` + `overtime_workers_count > 10` $\rightarrow$ Forces `safety_risk_score` >= `0.85`.
3. **The Communication Breakdown Paradox (`MODERATE_RISK`):** `communication_failure_count > 3` + `aircraft_on_ramp_count > 18` $\rightarrow$ Forces `safety_risk_score` >= `0.60` (Bypasses low-risk baselines due to uncoordinated blind movements).
4. **The Weekend Volume Catalyst (`MODERATE_RISK`):** `day_traffic_profile == "WEEKEND_RUSH"` + `aircraft_on_ramp_count > 20` $\rightarrow$ Standard schedule compression shifts risk into the `0.60 - 0.69` band.