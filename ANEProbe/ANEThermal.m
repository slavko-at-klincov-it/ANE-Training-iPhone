// ANEThermal.m — Thermal management and adaptive training for sustained ANE workloads
// Monitors ProcessInfo.thermalState, tracks ANE eval throughput via rolling window,
// and provides adaptive step delays to keep the device in safe thermal range.
#import "ANEThermal.h"
#import "ANETrainingConfig.h"
#import <UIKit/UIKit.h>

#pragma mark - Thermal State Monitoring

ThermalLevel current_thermal_level(void) {
    NSProcessInfoThermalState state = [[NSProcessInfo processInfo] thermalState];
    switch (state) {
        case NSProcessInfoThermalStateNominal:  return ThermalNominal;
        case NSProcessInfoThermalStateFair:     return ThermalFair;
        case NSProcessInfoThermalStateSerious:  return ThermalSerious;
        case NSProcessInfoThermalStateCritical: return ThermalCritical;
    }
    return ThermalNominal;
}

const char *thermal_level_name(ThermalLevel level) {
    switch (level) {
        case ThermalNominal:  return "nominal";
        case ThermalFair:     return "fair";
        case ThermalSerious:  return "serious";
        case ThermalCritical: return "critical";
    }
    return "unknown";
}

#pragma mark - ANE Throughput Monitor

// Rolling window of eval times
static double g_eval_times[THERMAL_WINDOW_SIZE];
static int    g_eval_count = 0;       // total evals recorded
static int    g_eval_idx   = 0;       // circular buffer index
static double g_baseline_sum = 0.0;   // sum of first THERMAL_WINDOW_SIZE samples
static int    g_baseline_n   = 0;     // how many baseline samples collected
static bool   g_baseline_locked = false;

void thermal_reset_monitor(void) {
    memset(g_eval_times, 0, sizeof(g_eval_times));
    g_eval_count = 0;
    g_eval_idx = 0;
    g_baseline_sum = 0.0;
    g_baseline_n = 0;
    g_baseline_locked = false;
}

void thermal_record_eval(double ms) {
    g_eval_times[g_eval_idx] = ms;
    g_eval_idx = (g_eval_idx + 1) % THERMAL_WINDOW_SIZE;
    g_eval_count++;

    // Collect baseline from the first THERMAL_WINDOW_SIZE samples
    if (!g_baseline_locked) {
        g_baseline_sum += ms;
        g_baseline_n++;
        if (g_baseline_n >= THERMAL_WINDOW_SIZE) {
            g_baseline_locked = true;
        }
    }
}

double thermal_baseline_avg(void) {
    if (g_baseline_n == 0) return 0.0;
    return g_baseline_sum / g_baseline_n;
}

double thermal_rolling_avg(void) {
    int n = g_eval_count < THERMAL_WINDOW_SIZE ? g_eval_count : THERMAL_WINDOW_SIZE;
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += g_eval_times[i];
    return sum / n;
}

bool thermal_is_throttled(void) {
    double base = thermal_baseline_avg();
    if (base <= 0.0) return false;  // not enough data
    double current = thermal_rolling_avg();
    return current > base * 2.0;    // >2x slowdown = throttled
}

#pragma mark - Adaptive Training Controller

ThermalConfig thermal_default_config(void) {
    return (ThermalConfig){
        .delay_nominal_ms   = 0,
        .delay_fair_ms      = 10,
        .delay_serious_ms   = 100,
        .pause_on_critical  = true,
        .cooldown_steps     = 5,
    };
}

// Internal cooldown counter (decremented by thermal_cooldown_tick)
static int g_cooldown_remaining = 0;

int thermal_step_delay(ThermalConfig *cfg) {
    ThermalLevel level = current_thermal_level();

    // If we're in cooldown after a thermal event, use serious-level delay
    if (g_cooldown_remaining > 0) {
        return cfg->delay_serious_ms;
    }

    // Throughput-based throttle detection: if ANE is throttling even at nominal/fair
    // thermal state, treat it as if we're at serious level
    if (thermal_is_throttled() && level < ThermalSerious) {
        level = ThermalSerious;
    }

    switch (level) {
        case ThermalNominal:
            return cfg->delay_nominal_ms;
        case ThermalFair:
            return cfg->delay_fair_ms;
        case ThermalSerious:
            g_cooldown_remaining = cfg->cooldown_steps;
            return cfg->delay_serious_ms;
        case ThermalCritical:
            g_cooldown_remaining = cfg->cooldown_steps;
            if (cfg->pause_on_critical) return -1;  // signal: pause entirely
            return cfg->delay_serious_ms;
    }
    return 0;
}

bool thermal_cooldown_tick(ThermalConfig *cfg) {
    (void)cfg;
    if (g_cooldown_remaining > 0) {
        g_cooldown_remaining--;
        return (g_cooldown_remaining == 0);
    }
    return true;  // already cooled down
}

#pragma mark - Power Monitoring

void thermal_enable_battery_monitoring(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [UIDevice currentDevice].batteryMonitoringEnabled = YES;
    });
}

void thermal_disable_battery_monitoring(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [UIDevice currentDevice].batteryMonitoringEnabled = NO;
    });
}

float thermal_battery_level(void) {
    return [UIDevice currentDevice].batteryLevel;
}

#pragma mark - ANE Device Info Power Drain Check

/// Check _ANEDeviceInfo.isExcessivePowerDrainWhenIdle if available
static bool ane_excessive_power_drain(void) {
    Class devInfoClass = NSClassFromString(@"_ANEDeviceInfo");
    if (!devInfoClass) return false;

    // Try to get shared instance or create one
    SEL sharedSel = NSSelectorFromString(@"sharedDeviceInfo");
    if (![devInfoClass respondsToSelector:sharedSel]) {
        // Try alloc/init
        id info = [[devInfoClass alloc] init];
        if (!info) return false;
        SEL drainSel = NSSelectorFromString(@"isExcessivePowerDrainWhenIdle");
        if (![info respondsToSelector:drainSel]) return false;
        return ((BOOL(*)(id,SEL))objc_msgSend)(info, drainSel);
    }
    id info = ((id(*)(Class,SEL))objc_msgSend)(devInfoClass, sharedSel);
    if (!info) return false;
    SEL drainSel = NSSelectorFromString(@"isExcessivePowerDrainWhenIdle");
    if (![info respondsToSelector:drainSel]) return false;
    return ((BOOL(*)(id,SEL))objc_msgSend)(info, drainSel);
}

#pragma mark - Thermal Stress Test

NSString *ane_thermal_test(void) {
    @autoreleasepool {
        ane_init();
        NSMutableString *out = [NSMutableString string];

        if (!g_D || !g_I) {
            [out appendString:@"  ANE classes missing\n"];
            return out;
        }

        [out appendString:@"  --- Thermal Stress Test ---\n"];
        [out appendString:@"  60-second sustained ANE workload\n"];
        [out appendString:@"  Kernel: 512x512 conv (medium load)\n\n"];

        // Enable battery monitoring
        thermal_enable_battery_monitoring();
        // Give main runloop a chance to process the battery enable
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
        float batteryStart = thermal_battery_level();

        // Reset throughput monitor
        thermal_reset_monitor();

        // Compile a medium kernel: 512-channel → 512-channel 1x1 conv
        int CH = 512, SQ = 64;
        float *weights = (float *)calloc(CH * CH, sizeof(float));
        srand48(99);
        float scale = 1.0f / sqrtf((float)CH);
        for (int i = 0; i < CH * CH; i++) weights[i] = (float)(2 * drand48() - 1) * scale;

        NSString *mil = [NSString stringWithFormat:
            @"%@    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
            @CONV_CONST
            "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
            "        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W,x=x)"
            "[name=string(\"out\")];\n"
            "    } -> (y);\n}\n",
            MIL_HDR, CH, SQ, CH, CH, CH, CH, CH, SQ];

        NSData *wBlob = build_blob(weights, CH, CH);
        NSDictionary *wDict = @{@"@model_path/weights/w.bin": @{@"offset":@0, @"data":wBlob}};

        [out appendString:@"  Compiling kernel...\n"];
        uint64_t tc = mach_absolute_time();
        Kern *k = compile_kern(mil, wDict, CH * SQ * 2, CH * SQ * 2);
        double compileMs = tb_ms(mach_absolute_time() - tc);
        free(weights);

        if (!k) {
            [out appendString:@"  COMPILE FAILED\n"];
            return out;
        }
        [out appendFormat:@"  Compile: %.1f ms\n\n", compileMs];

        // Fill input with random data
        float *inputData = (float *)calloc(CH * SQ, sizeof(float));
        for (int i = 0; i < CH * SQ; i++) inputData[i] = (float)(2 * drand48() - 1);
        io_write_fp16(k->ioIn, inputData, CH, SQ);
        free(inputData);

        // Header for per-second log
        [out appendString:@"   Sec  Evals  Avg(ms)  Thermal      Throttled  PowerDrain\n"];
        [out appendString:@"  ─────────────────────────────────────────────────────────\n"];

        // Run for 60 seconds, logging every second
        double testDuration = 60.0;
        int totalEvals = 0;
        double firstSecAvg = 0.0;
        double peakThroughput = 0.0;  // evals/sec
        double throttledThroughput = 0.0;
        double timeToThrottle = -1.0;
        bool everThrottled = false;

        // Per-second accumulators
        double secStart = 0.0;
        int secEvals = 0;
        double secSum = 0.0;

        uint64_t testStart = mach_absolute_time();

        for (int sec = 0; sec < (int)testDuration; sec++) {
            secEvals = 0;
            secSum = 0.0;
            secStart = tb_ms(mach_absolute_time() - testStart);

            // Run evals for ~1 second
            uint64_t secStartTime = mach_absolute_time();
            while (tb_ms(mach_absolute_time() - secStartTime) < 1000.0) {
                uint64_t evalStart = mach_absolute_time();
                ane_eval(k);
                double evalMs = tb_ms(mach_absolute_time() - evalStart);

                thermal_record_eval(evalMs);
                secSum += evalMs;
                secEvals++;
                totalEvals++;
            }

            double avgMs = secEvals > 0 ? secSum / secEvals : 0.0;
            double evalsPerSec = secEvals;
            ThermalLevel level = current_thermal_level();
            bool throttled = thermal_is_throttled();
            bool powerDrain = ane_excessive_power_drain();

            // Track stats
            if (sec == 0) firstSecAvg = avgMs;
            if (evalsPerSec > peakThroughput) peakThroughput = evalsPerSec;
            if (throttled && !everThrottled) {
                timeToThrottle = (double)sec + 1.0;
                everThrottled = true;
            }
            if (throttled || level >= ThermalSerious) {
                throttledThroughput = evalsPerSec;
            }

            [out appendFormat:@"  %4d  %5d  %6.2f   %-10s   %s         %s\n",
                sec + 1, secEvals, avgMs,
                thermal_level_name(level),
                throttled ? "YES" : "no ",
                powerDrain ? "YES" : "no"];
        }

        double totalMs = tb_ms(mach_absolute_time() - testStart);

        free_kern(k);

        // Battery at end
        float batteryEnd = thermal_battery_level();

        // Summary
        [out appendString:@"\n  ═══════════════════════════════════════════\n"];
        [out appendString:@"  THERMAL STRESS TEST SUMMARY\n"];
        [out appendString:@"  ═══════════════════════════════════════════\n"];
        [out appendFormat:@"  Duration:              %.1f sec\n", totalMs / 1000.0];
        [out appendFormat:@"  Total evals:           %d\n", totalEvals];
        [out appendFormat:@"  Avg evals/sec:         %.1f\n", totalEvals / (totalMs / 1000.0)];
        [out appendFormat:@"  Baseline eval time:    %.2f ms (first second avg)\n", firstSecAvg];
        [out appendFormat:@"  Peak throughput:       %.0f evals/sec\n", peakThroughput];
        [out appendFormat:@"  Final thermal state:   %s\n", thermal_level_name(current_thermal_level())];

        if (everThrottled) {
            [out appendFormat:@"  Time to throttle:      %.0f sec\n", timeToThrottle];
            [out appendFormat:@"  Throttled throughput:   %.0f evals/sec\n", throttledThroughput];
            double pct = (throttledThroughput / peakThroughput) * 100.0;
            [out appendFormat:@"  Throughput retention:   %.1f%%\n", pct];
        } else {
            [out appendString:@"  Throttling detected:   NO (device stayed cool)\n"];
        }

        [out appendFormat:@"  Rolling avg (end):     %.2f ms\n", thermal_rolling_avg()];
        [out appendFormat:@"  Baseline avg:          %.2f ms\n", thermal_baseline_avg()];

        if (batteryStart >= 0 && batteryEnd >= 0) {
            [out appendFormat:@"  Battery:               %.0f%% → %.0f%% (Δ%.1f%%)\n",
                batteryStart * 100, batteryEnd * 100, (batteryStart - batteryEnd) * 100];
        } else {
            [out appendString:@"  Battery:               monitoring unavailable\n"];
        }

        [out appendString:@"  ═══════════════════════════════════════════\n"];

        thermal_disable_battery_monitoring();
        return out;
    }
}
