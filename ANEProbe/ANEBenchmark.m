// ANEBenchmark.m — Performance, power, and efficiency benchmark for ANE training
// Measures throughput, power consumption, and training efficiency on iPhone ANE.
#import "ANEBenchmark.h"
#import "ANETrainingEngine.h"
#import "ANETrainingConfig.h"
#import "ANEThermal.h"
#import <UIKit/UIKit.h>
#import <mach/mach_time.h>
#import <mach/mach.h>
#include <sys/stat.h>
#include <sys/utsname.h>

// ===== Per-step timing record =====
typedef struct {
    int step;
    double step_ms;        // total step time
    float loss;
    bool is_compile_step;  // true if this step triggered Adam + recompile
    ThermalLevel thermal;
    float battery_level;   // 0.0-1.0, sampled every 10s
} BenchSample;

// ===== Aggregated stats =====
typedef struct {
    double sum, sum_sq;
    double min, max;
    int count;
} RunningStats;

static void stats_init(RunningStats *s) {
    s->sum = s->sum_sq = 0;
    s->min = 1e9; s->max = -1e9;
    s->count = 0;
}

static void stats_add(RunningStats *s, double val) {
    s->sum += val;
    s->sum_sq += val * val;
    if (val < s->min) s->min = val;
    if (val > s->max) s->max = val;
    s->count++;
}

static double stats_mean(RunningStats *s) {
    return s->count > 0 ? s->sum / s->count : 0;
}

static double stats_stddev(RunningStats *s) {
    if (s->count < 2) return 0;
    double mean = stats_mean(s);
    double var = (s->sum_sq / s->count) - mean * mean;
    return var > 0 ? sqrt(var) : 0;
}

// ===== Battery capacity estimates (Wh) for known devices =====
static double device_battery_wh(void) {
    // iPhone battery capacities (mAh * nominal voltage 3.83V / 1000)
    // These are approximate — actual varies with battery health
    struct utsname systemInfo;
    uname(&systemInfo);
    NSString *hw = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];

    // iPhone 15 Pro = 3274 mAh, 15 Pro Max = 4422 mAh
    // iPhone 16 Pro = 3582 mAh, 16 Pro Max = 4685 mAh
    // Default estimate: 3400 mAh (mid-range Pro)
    double mAh = 3400;

    if ([hw hasPrefix:@"iPhone16,1"])      mAh = 3274;  // 15 Pro
    else if ([hw hasPrefix:@"iPhone16,2"]) mAh = 4422;  // 15 Pro Max
    else if ([hw hasPrefix:@"iPhone17,1"]) mAh = 3582;  // 16 Pro
    else if ([hw hasPrefix:@"iPhone17,2"]) mAh = 4685;  // 16 Pro Max

    return mAh * 3.83 / 1000.0;  // Convert to Wh
}

// ===== Memory usage =====
static double resident_memory_mb(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        return (double)info.resident_size / (1024.0 * 1024.0);
    }
    return -1;
}

// ===== Main benchmark =====
NSString *ane_benchmark_run(float minutes) {
    @autoreleasepool {
    NSMutableString *out = [NSMutableString string];
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);

    #define TB_MS(t) ((double)(t) * tb.numer / tb.denom / 1e6)
    #define TB_S(t)  ((double)(t) * tb.numer / tb.denom / 1e9)

    [out appendString:@"╔═══════════════════════════════════════════════════════════╗\n"];
    [out appendString:@"║        ANE TRAINING BENCHMARK                            ║\n"];
    [out appendString:@"║        Performance · Power · Efficiency                   ║\n"];
    [out appendString:@"╚═══════════════════════════════════════════════════════════╝\n\n"];

    // --- Device info ---
    struct utsname systemInfo;
    uname(&systemInfo);
    NSString *hw = [NSString stringWithCString:systemInfo.machine encoding:NSUTF8StringEncoding];
    double batteryWh = device_battery_wh();

    [out appendFormat:@"  Device:          %@\n", hw];
    [out appendFormat:@"  Battery cap:     %.1f Wh (estimated)\n", batteryWh];
    [out appendFormat:@"  Model:           Stories-110M (%d layers, dim=%d, seq=%d)\n", NLAYERS, DIM, SEQ];
    [out appendFormat:@"  Accum steps:     %d (recompile every %d steps)\n", ACCUM_STEPS, ACCUM_STEPS];
    [out appendFormat:@"  Tokens/step:     %d\n", SEQ];
    [out appendFormat:@"  Duration:        %.0f min (%.1f min warmup + %.1f min measure)\n",
        minutes, fmin(2.0, minutes * 0.1), minutes - fmin(2.0, minutes * 0.1)];
    [out appendFormat:@"  Memory (start):  %.0f MB\n\n", resident_memory_mb()];

    // --- Enable battery monitoring ---
    thermal_enable_battery_monitoring();
    // Let main runloop process the battery enable
    [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.2]];
    float batteryStart = thermal_battery_level();
    [out appendFormat:@"  Battery start:   %.0f%%\n", batteryStart * 100];

    if (batteryStart < 0) {
        [out appendString:@"  WARNING: Battery monitoring unavailable (simulator?)\n"];
    }

    // --- Create dummy training data ---
    NSString *tmpPath = [NSTemporaryDirectory() stringByAppendingPathComponent:@"ane_bench_tokens.bin"];
    {
        size_t n = 200000;  // 200K tokens — enough for random sampling
        uint16_t *buf = (uint16_t *)malloc(n * 2);
        srand48(42);
        for (size_t i = 0; i < n; i++) buf[i] = (uint16_t)(drand48() * (VOCAB - 1));
        NSData *d = [NSData dataWithBytesNoCopy:buf length:n * 2 freeWhenDone:YES];
        [d writeToFile:tmpPath atomically:YES];
    }

    // --- Init training engine ---
    [out appendString:@"\n  Initializing training engine...\n"];
    uint64_t t_init = mach_absolute_time();
    ANETrainState *s = ane_train_init(NULL, [tmpPath UTF8String]);
    double init_ms = TB_MS(mach_absolute_time() - t_init);

    if (!s) {
        [out appendString:@"  FAIL: ane_train_init returned NULL\n"];
        return out;
    }
    [out appendFormat:@"  Init:            %.1fs\n", init_ms / 1000];
    [out appendFormat:@"  Memory (loaded): %.0f MB\n", resident_memory_mb()];

    // --- Timing setup ---
    double warmup_s = fmin(120.0, minutes * 60.0 * 0.1);  // 10% of total, max 2 min
    double target_s = minutes * 60.0;
    uint64_t bench_start = mach_absolute_time();

    // Stats accumulators
    RunningStats st_regular, st_compile, st_all;
    stats_init(&st_regular);
    stats_init(&st_compile);
    stats_init(&st_all);

    // Thermal timeline (sample every 10 seconds)
    #define MAX_THERMAL_SAMPLES 1024
    typedef struct {
        double elapsed_s;
        ThermalLevel level;
        float battery;
        double steps_per_sec;
        double avg_step_ms;
    } ThermalSample;
    ThermalSample thermal_timeline[MAX_THERMAL_SAMPLES];
    int thermal_sample_count = 0;
    double last_thermal_sample = 0;

    // Per-minute stats
    #define MAX_MINUTES 120
    typedef struct {
        int steps;
        double total_ms;
        float first_loss, last_loss, best_loss;
        int compile_steps;
        ThermalLevel worst_thermal;
    } MinuteStats;
    MinuteStats per_min[MAX_MINUTES];
    memset(per_min, 0, sizeof(per_min));
    for (int i = 0; i < MAX_MINUTES; i++) per_min[i].best_loss = 999;

    // Loss tracking
    float first_loss = 0, best_loss = 999, last_loss = 0;
    int total_steps = 0, compile_steps = 0;
    bool warmup_done = false;
    int warmup_steps = 0;

    // CSV log
    NSString *docsDir = [NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES) firstObject];
    NSString *csvPath = [docsDir stringByAppendingPathComponent:@"ane_benchmark.csv"];
    FILE *csv = fopen([csvPath UTF8String], "w");
    if (csv) {
        fprintf(csv, "step,step_ms,loss,is_compile,thermal,battery,elapsed_s,phase\n");
    }

    [out appendString:@"\n  ─── WARMUP PHASE ─── (stabilizing ANE throughput)\n"];
    fprintf(stderr, "BENCH: Starting benchmark (%.0f min)...\n", minutes);

    // ═══════════════════════════════════════════════
    // MAIN BENCHMARK LOOP
    // ═══════════════════════════════════════════════
    while (true) {
        double elapsed = TB_S(mach_absolute_time() - bench_start);
        if (elapsed >= target_s) break;

        // --- Run one training step ---
        uint64_t t_step = mach_absolute_time();
        float loss = ane_train_step(s);
        double step_ms = TB_MS(mach_absolute_time() - t_step);

        // Detect compile steps: every ACCUM_STEPS, Adam update triggers recompile.
        // These steps are significantly slower (~5x). We detect by step count modulo.
        int cur_step = ane_train_current_step(s);
        bool is_compile = (cur_step % ACCUM_STEPS == 0) && (cur_step > 0);

        total_steps++;
        last_loss = loss;
        if (loss < best_loss) best_loss = loss;
        if (total_steps == 1) first_loss = loss;

        // --- Warmup → Measure transition ---
        if (!warmup_done && elapsed >= warmup_s) {
            warmup_done = true;
            warmup_steps = total_steps;
            // Reset measurement stats
            stats_init(&st_regular);
            stats_init(&st_compile);
            stats_init(&st_all);
            compile_steps = 0;
            first_loss = loss;
            best_loss = loss;

            [out appendFormat:@"  Warmup done: %d steps in %.1fs (%.1f steps/s)\n",
                warmup_steps, elapsed, warmup_steps / elapsed];
            [out appendString:@"\n  ─── MEASURE PHASE ─── (collecting benchmark data)\n"];
            fprintf(stderr, "BENCH: Warmup done (%d steps). Measuring...\n", warmup_steps);
        }

        // --- Accumulate stats (only after warmup) ---
        if (warmup_done) {
            stats_add(&st_all, step_ms);
            if (is_compile) {
                stats_add(&st_compile, step_ms);
                compile_steps++;
            } else {
                stats_add(&st_regular, step_ms);
            }

            // Per-minute bucket
            double measure_elapsed = elapsed - warmup_s;
            int min_idx = (int)(measure_elapsed / 60.0);
            if (min_idx >= 0 && min_idx < MAX_MINUTES) {
                MinuteStats *m = &per_min[min_idx];
                m->steps++;
                m->total_ms += step_ms;
                m->last_loss = loss;
                if (m->steps == 1) m->first_loss = loss;
                if (loss < m->best_loss) m->best_loss = loss;
                if (is_compile) m->compile_steps++;
                ThermalLevel tl = current_thermal_level();
                if (tl > m->worst_thermal) m->worst_thermal = tl;
            }
        }

        // --- Thermal + battery sampling (every 10 seconds) ---
        if (elapsed - last_thermal_sample >= 10.0 && thermal_sample_count < MAX_THERMAL_SAMPLES) {
            ThermalSample *ts = &thermal_timeline[thermal_sample_count++];
            ts->elapsed_s = elapsed;
            ts->level = current_thermal_level();
            ts->battery = thermal_battery_level();
            ts->steps_per_sec = (warmup_done && st_all.count > 0)
                ? 1000.0 / stats_mean(&st_all) : total_steps / elapsed;
            ts->avg_step_ms = (st_all.count > 0) ? stats_mean(&st_all) : step_ms;
            last_thermal_sample = elapsed;
        }

        // --- CSV log ---
        if (csv) {
            fprintf(csv, "%d,%.2f,%.6f,%d,%d,%.4f,%.1f,%s\n",
                total_steps, step_ms, loss, is_compile ? 1 : 0,
                (int)current_thermal_level(),
                thermal_battery_level(), elapsed,
                warmup_done ? "measure" : "warmup");
            if (total_steps % 50 == 0) fflush(csv);
        }

        // --- Console progress (every 100 steps) ---
        if (total_steps % 100 == 0) {
            fprintf(stderr, "BENCH: step %d  loss=%.4f  %.1fms/step  thermal=%s  elapsed=%.0fs/%.0fs\n",
                total_steps, loss, step_ms,
                thermal_level_name(current_thermal_level()),
                elapsed, target_s);
        }
    }

    if (csv) fclose(csv);

    double total_elapsed = TB_S(mach_absolute_time() - bench_start);

    // --- Battery at end ---
    float batteryEnd = thermal_battery_level();
    double batteryDelta = (batteryStart >= 0 && batteryEnd >= 0)
        ? (batteryStart - batteryEnd) : -1;

    // ═══════════════════════════════════════════════
    // RESULTS
    // ═══════════════════════════════════════════════

    int measure_steps = st_all.count;
    double measure_s = total_elapsed - warmup_s;

    [out appendString:@"\n"];
    [out appendString:@"╔═══════════════════════════════════════════════════════════╗\n"];
    [out appendString:@"║                    BENCHMARK RESULTS                      ║\n"];
    [out appendString:@"╚═══════════════════════════════════════════════════════════╝\n\n"];

    // --- 1. PERFORMANCE ---
    [out appendString:@"  ┌─── PERFORMANCE ────────────────────────────────────────┐\n"];
    [out appendFormat:@"  │ Total steps:          %d (%d warmup + %d measured)     \n",
        total_steps, warmup_steps, measure_steps];
    [out appendFormat:@"  │ Duration:             %.1f min (%.1f min measured)     \n",
        total_elapsed / 60, measure_s / 60];
    [out appendFormat:@"  │                                                        \n"];
    [out appendFormat:@"  │ Overall:              %.2f steps/s                     \n",
        measure_steps / measure_s];
    [out appendFormat:@"  │ Tokens throughput:    %.0f tokens/s                    \n",
        (measure_steps * SEQ) / measure_s];
    [out appendFormat:@"  │                                                        \n"];
    [out appendFormat:@"  │ Regular steps:        avg %.1f ms (σ=%.1f, n=%d)      \n",
        stats_mean(&st_regular), stats_stddev(&st_regular), st_regular.count];
    [out appendFormat:@"  │                       min %.1f ms, max %.1f ms        \n",
        st_regular.min, st_regular.max];
    [out appendFormat:@"  │ Compile steps:        avg %.1f ms (σ=%.1f, n=%d)      \n",
        stats_mean(&st_compile), stats_stddev(&st_compile), st_compile.count];
    [out appendFormat:@"  │                       min %.1f ms, max %.1f ms        \n",
        st_compile.min, st_compile.max];
    [out appendFormat:@"  │ Compile overhead:     %.1fx slower                    \n",
        stats_mean(&st_compile) / fmax(stats_mean(&st_regular), 0.001)];
    [out appendFormat:@"  │ Compile ratio:        %d/%d steps (%.1f%%)            \n",
        compile_steps, measure_steps,
        measure_steps > 0 ? compile_steps * 100.0 / measure_steps : 0];
    [out appendFormat:@"  │ Adam updates:         %d                              \n",
        ane_train_current_step(s) / ACCUM_STEPS];
    [out appendFormat:@"  │ Memory (peak):        %.0f MB                         \n", resident_memory_mb()];
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // --- 2. POWER ---
    [out appendString:@"  ┌─── POWER CONSUMPTION ──────────────────────────────────┐\n"];
    if (batteryDelta > 0) {
        double energyWh = batteryDelta * batteryWh;
        double avgWatts = energyWh / (total_elapsed / 3600.0);
        double measureWatts = energyWh / (total_elapsed / 3600.0);  // same since battery is total

        [out appendFormat:@"  │ Battery drain:        %.1f%% (%.0f%% → %.0f%%)         \n",
            batteryDelta * 100, batteryStart * 100, batteryEnd * 100];
        [out appendFormat:@"  │ Energy consumed:      %.3f Wh                         \n", energyWh];
        [out appendFormat:@"  │ Avg power draw:       %.2f W                          \n", avgWatts];
        [out appendFormat:@"  │ Note: includes display + system overhead              \n"];

        // Confidence assessment
        if (batteryDelta >= 0.10) {
            [out appendString:@"  │ Confidence:           HIGH (>10% drain)               \n"];
        } else if (batteryDelta >= 0.05) {
            [out appendString:@"  │ Confidence:           MEDIUM (5-10% drain)            \n"];
        } else {
            [out appendString:@"  │ Confidence:           LOW (<5% drain, run longer)     \n"];
        }
    } else {
        [out appendString:@"  │ Battery monitoring:   unavailable                      \n"];
        [out appendString:@"  │ (run on device, not simulator)                         \n"];
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // --- 3. EFFICIENCY ---
    [out appendString:@"  ┌─── TRAINING EFFICIENCY ────────────────────────────────┐\n"];
    [out appendFormat:@"  │ Loss:                 %.4f → %.4f (best: %.4f)       \n",
        first_loss, last_loss, best_loss];
    double loss_reduction = (first_loss - best_loss) / first_loss * 100;
    [out appendFormat:@"  │ Loss reduction:       %.2f%%                           \n", loss_reduction];

    if (batteryDelta > 0) {
        double energyWh = batteryDelta * batteryWh;
        double tokensPerJoule = (measure_steps * SEQ) / (energyWh * 3600.0);
        double lossPerWh = (first_loss > best_loss) ? (first_loss - best_loss) / energyWh : 0;
        double stepsPerWh = measure_steps / energyWh;

        [out appendFormat:@"  │ Tokens per Joule:    %.1f                             \n", tokensPerJoule];
        [out appendFormat:@"  │ Steps per Wh:        %.0f                             \n", stepsPerWh];
        [out appendFormat:@"  │ Loss/Wh:             %.4f                             \n", lossPerWh];
        [out appendFormat:@"  │ Cost per 1%% loss:    %.3f Wh                         \n",
            loss_reduction > 0 ? energyWh / loss_reduction : 0];
    } else {
        double tokensPerSec = (measure_steps * SEQ) / measure_s;
        [out appendFormat:@"  │ Tokens/sec:          %.0f                             \n", tokensPerSec];
        [out appendString:@"  │ (power metrics need device battery data)              \n"];
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // --- 4. THERMAL PROFILE ---
    [out appendString:@"  ┌─── THERMAL PROFILE ────────────────────────────────────┐\n"];

    // Count time in each thermal state
    double time_nominal = 0, time_fair = 0, time_serious = 0, time_critical = 0;
    for (int i = 0; i < thermal_sample_count; i++) {
        double dt = (i + 1 < thermal_sample_count)
            ? thermal_timeline[i+1].elapsed_s - thermal_timeline[i].elapsed_s
            : 10.0;  // last sample
        switch (thermal_timeline[i].level) {
            case ThermalNominal:  time_nominal  += dt; break;
            case ThermalFair:     time_fair     += dt; break;
            case ThermalSerious:  time_serious  += dt; break;
            case ThermalCritical: time_critical += dt; break;
        }
    }
    double time_total = time_nominal + time_fair + time_serious + time_critical;
    if (time_total < 1) time_total = 1;

    [out appendFormat:@"  │ Nominal:    %5.0fs (%4.1f%%)                            \n",
        time_nominal, time_nominal / time_total * 100];
    [out appendFormat:@"  │ Fair:       %5.0fs (%4.1f%%)                            \n",
        time_fair, time_fair / time_total * 100];
    [out appendFormat:@"  │ Serious:    %5.0fs (%4.1f%%)                            \n",
        time_serious, time_serious / time_total * 100];
    [out appendFormat:@"  │ Critical:   %5.0fs (%4.1f%%)                            \n",
        time_critical, time_critical / time_total * 100];

    // Find when thermal state first changed
    for (int i = 1; i < thermal_sample_count; i++) {
        if (thermal_timeline[i].level != thermal_timeline[0].level) {
            [out appendFormat:@"  │ First thermal change: %.0fs (%s → %s)             \n",
                thermal_timeline[i].elapsed_s,
                thermal_level_name(thermal_timeline[i-1].level),
                thermal_level_name(thermal_timeline[i].level)];
            break;
        }
    }
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // --- 5. PER-MINUTE TIMELINE ---
    int total_min = (int)(measure_s / 60.0) + 1;
    if (total_min > MAX_MINUTES) total_min = MAX_MINUTES;
    if (total_min > 1) {
        [out appendString:@"  ┌─── PER-MINUTE TIMELINE ────────────────────────────────┐\n"];
        [out appendString:@"  │ Min  Steps  Avg(ms)  Loss     Steps/s  Compiles Thermal│\n"];
        [out appendString:@"  │ ─── ────── ──────── ──────── ──────── ──────── ────────│\n"];

        for (int i = 0; i < total_min; i++) {
            MinuteStats *m = &per_min[i];
            if (m->steps == 0) continue;
            double avg_ms = m->total_ms / m->steps;
            double sps = m->steps / 60.0;  // steps per second (approx)
            if (i == total_min - 1) {
                double frac = fmod(measure_s, 60.0);
                if (frac > 0) sps = m->steps / frac;
            }
            [out appendFormat:@"  │ %3d  %5d  %7.1f  %.4f  %6.1f   %5d    %-8s│\n",
                i + 1, m->steps, avg_ms, m->last_loss, sps, m->compile_steps,
                thermal_level_name(m->worst_thermal)];
        }
        [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];
    }

    // --- 6. DETAILED THERMAL TIMELINE (every 10s) ---
    if (thermal_sample_count > 2) {
        [out appendString:@"  ┌─── THERMAL TIMELINE (10s intervals) ───────────────────┐\n"];
        [out appendString:@"  │ Time    Thermal    Battery  Steps/s  Avg(ms)           │\n"];
        [out appendString:@"  │ ─────── ────────── ──────── ──────── ─────────         │\n"];

        // Show every sample for short runs, every Nth for long runs
        int skip = 1;
        if (thermal_sample_count > 60) skip = thermal_sample_count / 30;

        for (int i = 0; i < thermal_sample_count; i += skip) {
            ThermalSample *ts = &thermal_timeline[i];
            [out appendFormat:@"  │ %5.0fs  %-10s %5.1f%%   %5.1f    %7.1f           │\n",
                ts->elapsed_s,
                thermal_level_name(ts->level),
                ts->battery * 100,
                ts->steps_per_sec,
                ts->avg_step_ms];
        }
        // Always show last sample
        if (thermal_sample_count > 1) {
            ThermalSample *ts = &thermal_timeline[thermal_sample_count - 1];
            [out appendFormat:@"  │ %5.0fs  %-10s %5.1f%%   %5.1f    %7.1f  (final)   │\n",
                ts->elapsed_s,
                thermal_level_name(ts->level),
                ts->battery * 100,
                ts->steps_per_sec,
                ts->avg_step_ms];
        }
        [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];
    }

    // --- 7. KEY FINDINGS ---
    [out appendString:@"  ┌─── KEY FINDINGS ────────────────────────────────────────┐\n"];

    // Throughput stability: compare first vs last minute
    if (total_min >= 2) {
        MinuteStats *first = &per_min[0];
        MinuteStats *last_m = &per_min[total_min - 1];
        // Find last non-empty minute
        for (int i = total_min - 1; i >= 0; i--) {
            if (per_min[i].steps > 0) { last_m = &per_min[i]; break; }
        }
        if (first->steps > 0 && last_m->steps > 0) {
            double first_sps = first->steps / (first->total_ms / 1000.0);
            double last_sps = last_m->steps / (last_m->total_ms / 1000.0);
            double retention = last_sps / first_sps * 100;
            [out appendFormat:@"  │ Throughput retention: %.1f%% (%.1f → %.1f steps/s)  \n",
                retention, first_sps, last_sps];
            if (retention < 80) {
                [out appendString:@"  │ ⚠ Significant thermal throttling detected          \n"];
            } else if (retention < 95) {
                [out appendString:@"  │ Minor thermal impact on throughput                  \n"];
            } else {
                [out appendString:@"  │ Stable throughput (no thermal throttling)           \n"];
            }
        }
    }

    // Compile overhead impact
    if (st_compile.count > 0 && st_regular.count > 0) {
        double compile_pct = compile_steps * 100.0 / measure_steps;
        double time_in_compile = compile_steps * stats_mean(&st_compile);
        double time_in_regular = st_regular.count * stats_mean(&st_regular);
        double compile_time_pct = time_in_compile / (time_in_compile + time_in_regular) * 100;

        [out appendFormat:@"  │ Compile time budget:  %.1f%% of total step time       \n", compile_time_pct];
        [out appendFormat:@"  │ Without recompile:    %.1f steps/s (theoretical)      \n",
            1000.0 / stats_mean(&st_regular)];
        [out appendFormat:@"  │ With recompile:       %.1f steps/s (actual)           \n",
            measure_steps / measure_s];
    }

    [out appendString:@"  └────────────────────────────────────────────────────────┘\n\n"];

    // --- Data quality ---
    [out appendString:@"  ┌─── DATA QUALITY ────────────────────────────────────────┐\n"];
    [out appendFormat:@"  │ Regular step samples:  %d                              \n", st_regular.count];
    [out appendFormat:@"  │ Compile step samples:  %d                              \n", st_compile.count];
    [out appendFormat:@"  │ Thermal samples:       %d                              \n", thermal_sample_count];

    if (st_regular.count >= 100 && st_compile.count >= 20) {
        [out appendString:@"  │ Timing confidence:     HIGH                            \n"];
    } else if (st_regular.count >= 30) {
        [out appendString:@"  │ Timing confidence:     MEDIUM (run longer for better)  \n"];
    } else {
        [out appendString:@"  │ Timing confidence:     LOW (need more steps)           \n"];
    }

    if (batteryDelta >= 0.10) {
        [out appendString:@"  │ Power confidence:      HIGH (>10% drain)               \n"];
    } else if (batteryDelta >= 0.05) {
        [out appendString:@"  │ Power confidence:      MEDIUM (5-10% drain)            \n"];
    } else if (batteryDelta > 0) {
        [out appendFormat:@"  │ Power confidence:      LOW (%.1f%% drain, need 30+ min)\n",
            batteryDelta * 100];
    } else {
        [out appendString:@"  │ Power confidence:      N/A (no battery data)           \n"];
    }
    [out appendFormat:@"  │ CSV log:              %@\n", csvPath];
    [out appendString:@"  └────────────────────────────────────────────────────────┘\n"];

    // --- Cleanup ---
    ane_train_free(s);
    [[NSFileManager defaultManager] removeItemAtPath:tmpPath error:nil];
    thermal_disable_battery_monitoring();

    fprintf(stderr, "BENCH: Complete. %d steps in %.1f min.\n", total_steps, total_elapsed / 60);

    #undef TB_MS
    #undef TB_S

    return out;
    }
}
