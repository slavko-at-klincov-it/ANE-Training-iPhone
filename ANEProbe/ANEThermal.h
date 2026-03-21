// ANEThermal.h — Thermal management and adaptive training for sustained ANE workloads
// iPhone thermal constraints require monitoring + throttling adaptation to avoid
// degraded performance or system shutdown during long training runs.
#import <Foundation/Foundation.h>

// --- Thermal state monitoring ---
typedef enum {
    ThermalNominal  = 0,  // Full speed
    ThermalFair     = 1,  // Slightly warm, minor throttle
    ThermalSerious  = 2,  // Hot, significant throttle
    ThermalCritical = 3   // System at risk, must pause
} ThermalLevel;

/// Map NSProcessInfo.thermalState to our enum
ThermalLevel current_thermal_level(void);

/// Human-readable thermal state name
const char *thermal_level_name(ThermalLevel level);

// --- ANE throughput monitor (rolling window) ---
#define THERMAL_WINDOW_SIZE 32

/// Record an ANE eval time in milliseconds
void thermal_record_eval(double ms);

/// Returns true if recent eval times are >2x the baseline (first N samples)
bool thermal_is_throttled(void);

/// Get current rolling average eval time in ms (0 if no samples)
double thermal_rolling_avg(void);

/// Get baseline eval time in ms (average of first THERMAL_WINDOW_SIZE samples)
double thermal_baseline_avg(void);

/// Reset throughput monitor state
void thermal_reset_monitor(void);

// --- Adaptive training controller ---
typedef struct {
    int delay_nominal_ms;     // 0 — full speed
    int delay_fair_ms;        // 10 — slight cooldown
    int delay_serious_ms;     // 100 — significant cooldown
    bool pause_on_critical;   // true — stop completely until thermal drops
    int cooldown_steps;       // steps to skip after returning from serious/critical
} ThermalConfig;

/// Default config tuned for iPhone sustained ANE workloads
ThermalConfig thermal_default_config(void);

/// Returns ms to wait before next training step (or -1 if should pause entirely)
/// Also accounts for throughput-based throttle detection
int thermal_step_delay(ThermalConfig *cfg);

/// Track cooldown: call after each skipped step. Returns true when cooldown is done.
bool thermal_cooldown_tick(ThermalConfig *cfg);

// --- Power monitoring ---

/// Log battery level (0.0–1.0, or -1 if monitoring not enabled)
float thermal_battery_level(void);

/// Enable battery monitoring (must call before thermal_battery_level)
void thermal_enable_battery_monitoring(void);

/// Disable battery monitoring
void thermal_disable_battery_monitoring(void);

// --- Thermal stress test ---

/// Run a 60-second sustained ANE workload, logging thermal state and throughput.
/// Returns human-readable report string (like other test functions).
NSString *ane_thermal_test(void);
